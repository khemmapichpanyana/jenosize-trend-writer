"""Corpus store: `training_articles` in Postgres, accessed with psycopg.

Why a direct Postgres connection rather than the Supabase REST client the API
uses: the pipeline's hard problem is *incremental* work ("which rows changed
since this stage last ran?"), and that is one SQL predicate per stage —
`cleaned_hash is distinct from content_hash`. Expressing it over REST means
fetching every row and filtering in Python. `DATABASE_URL` also works with any
Postgres, not just Supabase.

Every stage method either selects the rows that still need work or records a
stage's result together with the fingerprint it was computed from. Nothing is
ever deleted: failures are recorded in `error`, duplicates are flagged.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import asynccontextmanager
from types import MappingProxyType
from typing import Any
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from pipeline.schemas import ArticleLabels, Split, TrainingArticle

_COLUMNS = ", ".join(TrainingArticle.model_fields)


def _article(row: dict[str, Any]) -> TrainingArticle:
    data = {k: row.get(k) for k in TrainingArticle.model_fields}
    data["labels"] = ArticleLabels.model_validate(row["labels"]) if row.get("labels") else None
    data["split"] = data.get("split") or "train"
    data["word_count"] = data.get("word_count") or 0
    data["is_duplicate"] = bool(data.get("is_duplicate"))
    return TrainingArticle.model_validate(data)


# Corpus browsing filters: a fixed map, so no caller-supplied SQL ever reaches a
# query string.
ARTICLE_STATES: Mapping[str, str] = MappingProxyType(
    {
        "all": "true",
        "pending": "content_hash is null",
        "fetched": "content_hash is not null",
        "cleaned": "clean_markdown is not null and not is_duplicate",
        "rejected": "error is not null",
        "duplicates": "is_duplicate",
        "labelled": "labelled_hash is not null and labelled_hash = cleaned_hash",
    }
)


class CorpusStore:
    """Async Postgres access for the pipeline. Use via `connect()`."""

    def __init__(self, conn: psycopg.AsyncConnection[dict[str, Any]]) -> None:
        self._conn = conn

    # ------------------------------------------------------------- discover

    async def add_urls(self, articles: Iterable[TrainingArticle]) -> list[str]:
        """Insert URLs not seen before; return only the new ones.

        Known URLs are left untouched: rediscovering an article is not news.
        """
        rows = [(a.url, a.category_slug) for a in articles]
        if not rows:
            return []
        async with self._conn.cursor() as cur:
            await cur.executemany(
                "insert into training_articles (url, category_slug) values (%s, %s) "
                "on conflict (url) do nothing returning url",
                rows,
                returning=True,
            )
            new: list[str] = []
            while True:
                if (row := await cur.fetchone()) is not None:
                    new.append(row["url"])
                if not cur.nextset():
                    break
        return new

    # ---------------------------------------------------------------- crawl

    async def due_for_crawl(
        self, *, limit: int, recheck_days: float | None
    ) -> list[TrainingArticle]:
        """Pages never successfully checked, plus stale ones if asked.

        "Checked" is stamped on success and on *permanent* failures (a 4xx, or a
        page with no article text), so those are not retried every run; a
        transient failure (timeout, 5xx) leaves it unset and is retried next run.

        Articles are rarely edited after publication, so re-checking known pages
        is opt-in (`recheck_days`) rather than part of every run.
        """
        query = f"""
            select {_COLUMNS} from training_articles
            where last_checked_at is null
               or (%(recheck)s::float is not null
                   and last_checked_at < now() - make_interval(secs => %(recheck)s::float * 86400))
            order by last_checked_at nulls first, url
            limit %(limit)s
        """
        async with self._conn.cursor() as cur:
            await cur.execute(query, {"recheck": recheck_days, "limit": limit})
            return [_article(r) for r in await cur.fetchall()]

    async def current_hash(self, url: str) -> str | None:
        async with self._conn.cursor() as cur:
            await cur.execute("select content_hash from training_articles where url = %s", (url,))
            row = await cur.fetchone()
        return row["content_hash"] if row else None

    async def mark_unchanged(self, url: str) -> None:
        """Content identical to what we hold: note the check, write nothing else."""
        await self._conn.execute(
            "update training_articles set last_checked_at = now(), "
            "fetch_count = fetch_count + 1, error = null where url = %s",
            (url,),
        )

    async def record_content(
        self,
        url: str,
        *,
        content_hash: str,
        r2_raw_key: str,
        title: str | None,
        meta_description: str | None,
    ) -> None:
        """New or changed content. Downstream stages notice via the fingerprint."""
        await self._conn.execute(
            """
            update training_articles set
                content_hash = %s, r2_raw_key = %s,
                title = coalesce(%s, title), meta_description = coalesce(%s, meta_description),
                last_checked_at = now(), content_changed_at = now(),
                fetch_count = fetch_count + 1, error = null
            where url = %s
            """,
            (content_hash, r2_raw_key, title, meta_description, url),
        )

    async def record_error(self, url: str, message: str, *, checked: bool = False) -> None:
        # `checked` stamps last_checked_at so a permanently broken page is not
        # retried at the head of every queue; transient errors leave it unset.
        stamp = ", last_checked_at = now()" if checked else ""
        await self._conn.execute(
            f"update training_articles set error = %s{stamp} where url = %s",
            (message[:500], url),
        )

    # ---------------------------------------------------------------- clean

    async def due_for_clean(self, *, clean_version: int) -> list[TrainingArticle]:
        async with self._conn.cursor() as cur:
            await cur.execute(
                f"""
                select {_COLUMNS} from training_articles
                where content_hash is not null
                  and (cleaned_hash is distinct from content_hash
                       or clean_version is distinct from %s)
                order by url
                """,
                (clean_version,),
            )
            return [_article(r) for r in await cur.fetchall()]

    async def record_clean(
        self,
        url: str,
        *,
        content_hash: str,
        clean_version: int,
        clean_markdown: str | None,
        word_count: int,
        error: str | None,
    ) -> None:
        """Store the cleaning result, including a rejection (markdown = null).

        Recording rejections with their fingerprint means a too-short article
        is not re-cleaned every run; it only comes back if its content changes
        or the cleaning rules do.
        """
        await self._conn.execute(
            """
            update training_articles set
                clean_markdown = %s, word_count = %s, error = %s,
                cleaned_hash = %s, clean_version = %s
            where url = %s
            """,
            (clean_markdown, word_count, error, content_hash, clean_version, url),
        )

    async def cleaned(self) -> list[TrainingArticle]:
        async with self._conn.cursor() as cur:
            await cur.execute(
                f"select {_COLUMNS} from training_articles "
                "where clean_markdown is not null order by first_seen_at, url"
            )
            return [_article(r) for r in await cur.fetchall()]

    async def set_duplicates(self, duplicate_urls: set[str]) -> None:
        """Flag exactly this set as duplicates (and clear stale flags)."""
        await self._conn.execute(
            "update training_articles set is_duplicate = (url = any(%s)) "
            "where clean_markdown is not null",
            (list(duplicate_urls),),
        )

    # ---------------------------------------------------------------- label

    async def due_for_label(self, *, label_version: int) -> list[TrainingArticle]:
        async with self._conn.cursor() as cur:
            await cur.execute(
                f"""
                select {_COLUMNS} from training_articles
                where clean_markdown is not null and not is_duplicate
                  and (labelled_hash is distinct from cleaned_hash
                       or label_version is distinct from %s)
                order by url
                """,
                (label_version,),
            )
            return [_article(r) for r in await cur.fetchall()]

    async def record_labels(
        self,
        url: str,
        *,
        labels: ArticleLabels,
        labelled_hash: str,
        label_version: int,
        labeler_model: str,
    ) -> None:
        await self._conn.execute(
            """
            update training_articles set
                labels = %s, labelled_hash = %s, label_version = %s,
                labeler_model = %s, error = null
            where url = %s
            """,
            (
                Jsonb(labels.model_dump(mode="json")),
                labelled_hash,
                label_version,
                labeler_model,
                url,
            ),
        )

    # ---------------------------------------------------------------- build

    async def usable(self) -> list[TrainingArticle]:
        """Rows that can become training examples right now."""
        async with self._conn.cursor() as cur:
            await cur.execute(
                f"""
                select {_COLUMNS} from training_articles
                where labels is not null and clean_markdown is not null
                  and title is not null and not is_duplicate
                  and labelled_hash is not distinct from cleaned_hash
                order by url
                """
            )
            return [_article(r) for r in await cur.fetchall()]

    async def set_splits(self, splits: dict[str, Split]) -> None:
        async with self._conn.cursor() as cur:
            await cur.executemany(
                "update training_articles set split = %s where url = %s",
                [(split, url) for url, split in splits.items()],
            )

    async def dataset_versions(self) -> list[dict[str, Any]]:
        async with self._conn.cursor() as cur:
            await cur.execute("select * from dataset_versions order by created_at desc")
            return list(await cur.fetchall())

    async def dataset_version(self, version: str) -> dict[str, Any] | None:
        async with self._conn.cursor() as cur:
            await cur.execute("select * from dataset_versions where version = %s", (version,))
            return await cur.fetchone()

    async def record_dataset_version(self, *, replace: bool = False, **fields: Any) -> None:
        fields["params"] = Jsonb(fields.get("params") or {})
        columns = ", ".join(fields)
        placeholders = ", ".join(f"%({k})s" for k in fields)
        conflict = ""
        if replace:
            updates = ", ".join(f"{k} = excluded.{k}" for k in fields if k != "version")
            conflict = f" on conflict (version) do update set {updates}, created_at = now()"
        await self._conn.execute(
            f"insert into dataset_versions ({columns}) values ({placeholders}){conflict}",
            fields,
        )

    # ----------------------------------------------------------- browsing

    async def list_articles(
        self, *, state: str = "all", category: str | None = None, limit: int = 50, offset: int = 0
    ) -> tuple[int, list[TrainingArticle]]:
        """Page through the corpus by pipeline state (the predicate is a fixed map)."""
        where = ARTICLE_STATES[state]
        params = {"category": category, "limit": limit, "offset": offset}
        filters = f"({where}) and (%(category)s::text is null or category_slug = %(category)s)"
        async with self._conn.cursor() as cur:
            await cur.execute(
                f"select count(*) as n from training_articles where {filters}", params
            )
            total = int((await cur.fetchone() or {"n": 0})["n"])
            await cur.execute(
                f"select {_COLUMNS} from training_articles where {filters} "
                "order by url limit %(limit)s offset %(offset)s",
                params,
            )
            return total, [_article(r) for r in await cur.fetchall()]

    async def sample(self, *, n: int, labelled: bool, seed: float) -> list[TrainingArticle]:
        """A reproducible random sample of cleaned articles, for human review."""
        where = ARTICLE_STATES["labelled" if labelled else "cleaned"]
        async with self._conn.cursor() as cur:
            await cur.execute("select setseed(%s)", (seed,))
            await cur.execute(
                f"select {_COLUMNS} from training_articles where {where} order by random() limit %s",
                (n,),
            )
            return [_article(r) for r in await cur.fetchall()]

    # ----------------------------------------------------------- reporting

    async def get(self, url: str) -> TrainingArticle | None:
        async with self._conn.cursor() as cur:
            await cur.execute(f"select {_COLUMNS} from training_articles where url = %s", (url,))
            row = await cur.fetchone()
        return _article(row) if row else None

    async def all(self) -> list[TrainingArticle]:
        async with self._conn.cursor() as cur:
            await cur.execute(f"select {_COLUMNS} from training_articles order by url")
            return [_article(r) for r in await cur.fetchall()]

    async def counts(self, *, clean_version: int, label_version: int) -> dict[str, int]:
        async with self._conn.cursor() as cur:
            await cur.execute(
                """
                select
                  count(*)                                                    as discovered,
                  count(*) filter (where content_hash is not null)            as fetched,
                  count(*) filter (where clean_markdown is not null
                                   and cleaned_hash = content_hash
                                   and clean_version = %(cv)s)                as cleaned,
                  count(*) filter (where labels is not null
                                   and labelled_hash = cleaned_hash
                                   and label_version = %(lv)s)                as labelled,
                  count(*) filter (where is_duplicate)                        as duplicates,
                  count(*) filter (where error is not null)                   as errors
                from training_articles
                """,
                {"cv": clean_version, "lv": label_version},
            )
            row = await cur.fetchone()
        return {k: int(v or 0) for k, v in (row or {}).items()}

    # ----------------------------------------------------------- run log

    async def start_run(self, stage: str, *, job_id: UUID | None = None) -> UUID:
        async with self._conn.cursor() as cur:
            await cur.execute(
                "insert into pipeline_runs (stage, job_id) values (%s, %s) returning id",
                (stage, job_id),
            )
            row = await cur.fetchone()
        assert row is not None
        return UUID(str(row["id"]))

    async def finish_run(
        self, run_id: UUID, *, stats: dict[str, Any], error: str | None = None
    ) -> None:
        await self._conn.execute(
            "update pipeline_runs set status = %s, finished_at = now(), stats = %s, error = %s "
            "where id = %s",
            ("failed" if error else "succeeded", Jsonb(stats), error, run_id),
        )

    async def stage_runs_for_job(self, job_id: UUID) -> list[dict[str, Any]]:
        async with self._conn.cursor() as cur:
            await cur.execute(
                "select stage, status, started_at, finished_at, stats, error "
                "from pipeline_runs where job_id = %s order by started_at",
                (job_id,),
            )
            return list(await cur.fetchall())

    async def recent_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        async with self._conn.cursor() as cur:
            await cur.execute(
                "select stage, status, started_at, finished_at, stats, error "
                "from pipeline_runs order by started_at desc limit %s",
                (limit,),
            )
            return list(await cur.fetchall())


class ActiveJobExists(RuntimeError):
    """Another job of the same kind is queued or running."""

    def __init__(self, kind: str, existing: dict[str, Any] | None) -> None:
        super().__init__(f"a {kind} job is already active")
        self.kind = kind
        self.existing = existing


class JobStore:
    """`job_runs`: jobs started through the jobs API."""

    def __init__(self, conn: psycopg.AsyncConnection[dict[str, Any]]) -> None:
        self._conn = conn

    async def create(self, kind: str, params: dict[str, Any]) -> dict[str, Any]:
        """Insert a queued job; the partial unique index enforces one active per kind."""
        try:
            async with self._conn.cursor() as cur:
                await cur.execute(
                    "insert into job_runs (kind, params) values (%s, %s) returning *",
                    (kind, Jsonb(params)),
                )
                row = await cur.fetchone()
        except psycopg.errors.UniqueViolation:
            raise ActiveJobExists(kind, await self.active(kind)) from None
        assert row is not None
        return dict(row)

    async def active(self, kind: str) -> dict[str, Any] | None:
        async with self._conn.cursor() as cur:
            await cur.execute(
                "select * from job_runs where kind = %s and status in ('queued', 'running')",
                (kind,),
            )
            return await cur.fetchone()

    async def set_call_id(self, job_id: UUID, call_id: str) -> None:
        await self._conn.execute(
            "update job_runs set modal_call_id = %s where id = %s", (call_id, job_id)
        )

    async def mark_running(self, job_id: UUID) -> None:
        await self._conn.execute(
            "update job_runs set status = 'running', started_at = now() "
            "where id = %s and status = 'queued'",
            (job_id,),
        )

    async def finish(
        self,
        job_id: UUID,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Move to a terminal state. A job already terminal (e.g. cancelled) wins."""
        await self._conn.execute(
            """
            update job_runs set status = %s, result = %s, error = %s, finished_at = now()
            where id = %s and status in ('queued', 'running')
            """,
            (status, Jsonb(result) if result is not None else None, error, job_id),
        )

    async def get(self, job_id: UUID) -> dict[str, Any] | None:
        async with self._conn.cursor() as cur:
            await cur.execute("select * from job_runs where id = %s", (job_id,))
            return await cur.fetchone()

    async def list(self, *, limit: int = 20, kind: str | None = None) -> list[dict[str, Any]]:
        async with self._conn.cursor() as cur:
            await cur.execute(
                "select * from job_runs where (%s::text is null or kind = %s) "
                "order by created_at desc limit %s",
                (kind, kind, limit),
            )
            return list(await cur.fetchall())


@asynccontextmanager
async def connect_jobs(dsn: str) -> AsyncIterator[tuple[CorpusStore, JobStore]]:
    """Both stores over one connection, for the jobs API and job workers."""
    conn = await psycopg.AsyncConnection.connect(
        dsn, autocommit=True, row_factory=dict_row, prepare_threshold=None
    )
    try:
        yield CorpusStore(conn), JobStore(conn)
    finally:
        await conn.close()


@asynccontextmanager
async def connect(dsn: str) -> AsyncIterator[CorpusStore]:
    """Open one autocommitting connection for a CLI invocation.

    Autocommit because each stage records progress row by row: a crash halfway
    through a crawl must keep the pages already done, so the re-run resumes
    rather than starts over.

    `prepare_threshold=None` disables server-side prepared statements, which
    Supabase's transaction pooler (and PgBouncer generally) cannot route.
    """
    conn = await psycopg.AsyncConnection.connect(
        dsn, autocommit=True, row_factory=dict_row, prepare_threshold=None
    )
    try:
        yield CorpusStore(conn)
    finally:
        await conn.close()


def split_for(url: str, eval_frac: float, seed: int = 13) -> Split:
    """Deterministic train/eval assignment from the URL.

    Hash-based rather than random so that re-running `build_dataset` after adding
    new articles never moves an existing article across the split — which would
    silently leak eval data into a later training run.
    """
    import hashlib

    digest = hashlib.sha256(f"{seed}:{url}".encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket < eval_frac else "train"
