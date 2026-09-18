"""Corpus store — where the pipeline keeps its work between CLI invocations.

The API's `Repository` is request-shaped (one generation at a time); the pipeline
needs bulk scans and partial updates across four separate processes, so it gets
its own seam with the same local-first philosophy:

  * `PERSISTENCE=none`  -> SQLite at `.data/corpus.db`. Zero accounts, and the
    stages are resumable because state survives the process.
  * `PERSISTENCE=supabase` -> the `training_articles` table.

SQLite rather than JSONL because every stage does "read the rows that still need
work, update them by url", which is an index lookup, not a file rewrite.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import anyio

from app.core.config import Settings
from pipeline.schemas import ArticleLabels, Split, TrainingArticle

# Columns shared by both backends. Kept explicit so a schema drift between
# SQLite and Postgres shows up as a KeyError here rather than silent data loss.
_FIELDS = (
    "url",
    "category_slug",
    "title",
    "meta_description",
    "raw_key",
    "content_hash",
    "clean_markdown",
    "word_count",
    "is_duplicate",
    "labels",
    "split",
    "error",
    "fetched_at",
)


@runtime_checkable
class CorpusStore(Protocol):
    name: str

    async def upsert(self, article: TrainingArticle) -> None: ...

    async def upsert_many(self, articles: list[TrainingArticle]) -> int: ...

    async def get(self, url: str) -> TrainingArticle | None: ...

    async def all(self) -> list[TrainingArticle]: ...

    async def counts(self) -> dict[str, int]: ...


def _to_row(article: TrainingArticle) -> dict[str, Any]:
    row = article.model_dump(mode="json")
    row["labels"] = json.dumps(row["labels"]) if row.get("labels") else None
    return {key: row.get(key) for key in _FIELDS}


def _from_row(row: dict[str, Any]) -> TrainingArticle:
    data = dict(row)
    labels = data.get("labels")
    if isinstance(labels, str):
        labels = json.loads(labels) if labels else None
    data["labels"] = ArticleLabels.model_validate(labels) if labels else None
    data["is_duplicate"] = bool(data.get("is_duplicate"))
    return TrainingArticle.model_validate(data)


class SqliteCorpusStore:
    """Local, resumable corpus state."""

    name = "sqlite"

    def __init__(self, path: str = ".data/corpus.db") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                create table if not exists articles (
                    url              text primary key,
                    category_slug    text,
                    title            text,
                    meta_description text,
                    raw_key          text,
                    content_hash     text,
                    clean_markdown   text,
                    word_count       integer default 0,
                    is_duplicate     integer default 0,
                    labels           text,
                    split            text default 'train',
                    error            text,
                    fetched_at       text
                )
                """
            )
            conn.execute("create index if not exists articles_split on articles (split)")

    def _upsert_sync(self, articles: list[TrainingArticle]) -> int:
        rows = [_to_row(a) for a in articles]
        columns = ", ".join(_FIELDS)
        placeholders = ", ".join(f":{f}" for f in _FIELDS)
        # Only overwrite a column when the new value is non-null, so a later
        # stage re-running does not wipe an earlier stage's output.
        updates = ", ".join(f"{f} = coalesce(excluded.{f}, articles.{f})" for f in _FIELDS[1:])
        with self._conn() as conn:
            conn.executemany(
                f"insert into articles ({columns}) values ({placeholders}) "
                f"on conflict(url) do update set {updates}",
                rows,
            )
        return len(rows)

    async def upsert(self, article: TrainingArticle) -> None:
        await anyio.to_thread.run_sync(self._upsert_sync, [article])

    async def upsert_many(self, articles: list[TrainingArticle]) -> int:
        if not articles:
            return 0
        return await anyio.to_thread.run_sync(self._upsert_sync, articles)

    async def get(self, url: str) -> TrainingArticle | None:
        def _get() -> TrainingArticle | None:
            with self._conn() as conn:
                row = conn.execute("select * from articles where url = ?", (url,)).fetchone()
            return _from_row(dict(row)) if row else None

        return await anyio.to_thread.run_sync(_get)

    async def all(self) -> list[TrainingArticle]:
        def _all() -> list[TrainingArticle]:
            with self._conn() as conn:
                rows = conn.execute("select * from articles order by url").fetchall()
            return [_from_row(dict(r)) for r in rows]

        return await anyio.to_thread.run_sync(_all)

    async def counts(self) -> dict[str, int]:
        def _counts() -> dict[str, int]:
            with self._conn() as conn:
                one = conn.execute(
                    """
                    select
                      count(*)                                            as discovered,
                      sum(case when raw_key is not null then 1 else 0 end)        as scraped,
                      sum(case when clean_markdown is not null then 1 else 0 end) as cleaned,
                      sum(case when labels is not null then 1 else 0 end)         as labelled,
                      sum(case when is_duplicate then 1 else 0 end)               as duplicates,
                      sum(case when error is not null then 1 else 0 end)          as errors
                    from articles
                    """
                ).fetchone()
            # sqlite3.Row iterates values, not column names, so .keys() is required.
            return {k: int(one[k] or 0) for k in one.keys()}  # noqa: SIM118

        return await anyio.to_thread.run_sync(_counts)


class SupabaseCorpusStore:
    """`training_articles` in Postgres — the shared, durable corpus."""

    name = "supabase"

    def __init__(self, settings: Settings) -> None:
        from app.db.supabase_repo import SupabaseRepository

        self._repo = SupabaseRepository(settings)

    async def upsert(self, article: TrainingArticle) -> None:
        await self.upsert_many([article])

    async def upsert_many(self, articles: list[TrainingArticle]) -> int:
        if not articles:
            return 0
        rows = []
        for article in articles:
            row = _to_row(article)
            # Postgres holds `labels` as jsonb, so it takes the dict, not a string.
            row["labels"] = article.labels.model_dump(mode="json") if article.labels else None
            row["r2_raw_key"] = row.pop("raw_key")
            rows.append(row)

        def _upsert() -> None:
            self._repo._client.table("training_articles").upsert(rows, on_conflict="url").execute()

        await anyio.to_thread.run_sync(_upsert)
        return len(rows)

    async def get(self, url: str) -> TrainingArticle | None:
        def _get() -> dict[str, Any] | None:
            result = (
                self._repo._client.table("training_articles")
                .select("*")
                .eq("url", url)
                .limit(1)
                .execute()
            )
            data = result.data or []
            return data[0] if data else None

        row = await anyio.to_thread.run_sync(_get)
        return _from_row(_rename_pg(row)) if row else None

    async def all(self) -> list[TrainingArticle]:
        def _all() -> list[dict[str, Any]]:
            result = self._repo._client.table("training_articles").select("*").execute()
            return list(result.data or [])

        return [_from_row(_rename_pg(row)) for row in await anyio.to_thread.run_sync(_all)]

    async def counts(self) -> dict[str, int]:
        articles = await self.all()
        return {
            "discovered": len(articles),
            "scraped": sum(1 for a in articles if a.raw_key),
            "cleaned": sum(1 for a in articles if a.clean_markdown),
            "labelled": sum(1 for a in articles if a.labels),
            "duplicates": sum(1 for a in articles if a.is_duplicate),
            "errors": sum(1 for a in articles if a.error),
        }


def _rename_pg(row: dict[str, Any]) -> dict[str, Any]:
    """Postgres calls it `r2_raw_key`; the pipeline model calls it `raw_key`."""
    out = {k: v for k, v in row.items() if k in {*_FIELDS, "r2_raw_key"}}
    if "r2_raw_key" in out:
        out["raw_key"] = out.pop("r2_raw_key")
    return out


def build_store(settings: Settings) -> CorpusStore:
    if settings.persistence == "supabase":
        return SupabaseCorpusStore(settings)
    return SqliteCorpusStore(str(Path(settings.local_storage_dir) / "corpus.db"))


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
