"""CorpusStore against a real Postgres: the incremental predicates per stage."""

from __future__ import annotations

import pytest

from pipeline.schemas import ArticleLabels, TrainingArticle
from pipeline.store import CorpusStore, split_for

URL_A = "https://www.jenosize.com/en/ideas/futurist/a"
URL_B = "https://www.jenosize.com/en/ideas/futurist/b"


async def _fetched(corpus: CorpusStore, url: str = URL_A, content: str = "h1") -> None:
    await corpus.add_urls([TrainingArticle(url=url, category_slug="futurist")])
    await corpus.record_content(
        url, content_hash=content, r2_raw_key=f"raw/{content}.html", title="T", meta_description="M"
    )


async def test_add_urls_returns_only_new_urls(corpus: CorpusStore) -> None:
    first = await corpus.add_urls([TrainingArticle(url=URL_A), TrainingArticle(url=URL_B)])
    again = await corpus.add_urls([TrainingArticle(url=URL_A), TrainingArticle(url=URL_B)])
    assert sorted(first) == [URL_A, URL_B]
    assert again == []


async def test_add_urls_does_not_touch_known_rows(corpus: CorpusStore) -> None:
    await _fetched(corpus)
    await corpus.add_urls([TrainingArticle(url=URL_A, category_slug="changed")])
    row = await corpus.get(URL_A)
    assert row is not None and row.category_slug == "futurist" and row.content_hash == "h1"


async def test_new_urls_are_due_for_crawl_and_fetched_ones_are_not(corpus: CorpusStore) -> None:
    await corpus.add_urls([TrainingArticle(url=URL_A), TrainingArticle(url=URL_B)])
    await corpus.record_content(
        URL_A, content_hash="h", r2_raw_key="k", title=None, meta_description=None
    )
    due = [a.url for a in await corpus.due_for_crawl(limit=10, recheck_days=None)]
    assert due == [URL_B]


async def test_recheck_days_brings_back_stale_pages(corpus: CorpusStore) -> None:
    await _fetched(corpus)
    assert await corpus.due_for_crawl(limit=10, recheck_days=None) == []
    # recheck_days=0 means "anything checked before now", i.e. everything.
    assert [a.url for a in await corpus.due_for_crawl(limit=10, recheck_days=0)] == [URL_A]


async def test_transient_errors_are_retried_permanent_ones_are_not(corpus: CorpusStore) -> None:
    await corpus.add_urls([TrainingArticle(url=URL_A), TrainingArticle(url=URL_B)])
    await corpus.record_error(URL_A, "timeout")  # transient: stays unchecked
    await corpus.record_error(URL_B, "404", checked=True)  # permanent
    due = [a.url for a in await corpus.due_for_crawl(limit=10, recheck_days=None)]
    # Found live: an empty page marked permanent was being retried every run.
    assert due == [URL_A]
    # ...but a recheck still revisits it, in case the page was fixed.
    assert URL_B in [a.url for a in await corpus.due_for_crawl(limit=10, recheck_days=0)]


async def test_mark_unchanged_writes_nothing_but_the_check(corpus: CorpusStore) -> None:
    await _fetched(corpus)
    before = await corpus.get(URL_A)
    await corpus.mark_unchanged(URL_A)
    after = await corpus.get(URL_A)
    assert before is not None and after is not None
    assert after.content_hash == before.content_hash
    assert after.content_changed_at == before.content_changed_at
    assert after.last_checked_at is not None and before.last_checked_at is not None
    assert after.last_checked_at >= before.last_checked_at


async def test_clean_is_due_only_when_content_or_rules_change(corpus: CorpusStore) -> None:
    await _fetched(corpus, content="h1")
    assert [a.url for a in await corpus.due_for_clean(clean_version=1)] == [URL_A]

    await corpus.record_clean(
        URL_A, content_hash="h1", clean_version=1, clean_markdown="## x", word_count=400, error=None
    )
    assert await corpus.due_for_clean(clean_version=1) == []  # nothing new
    assert len(await corpus.due_for_clean(clean_version=2)) == 1  # rules changed

    await corpus.record_content(
        URL_A, content_hash="h2", r2_raw_key="k2", title=None, meta_description=None
    )
    assert len(await corpus.due_for_clean(clean_version=1)) == 1  # content changed


async def test_rejected_articles_are_not_recleaned_every_run(corpus: CorpusStore) -> None:
    await _fetched(corpus)
    await corpus.record_clean(
        URL_A,
        content_hash="h1",
        clean_version=1,
        clean_markdown=None,
        word_count=80,
        error="too short",
    )
    assert await corpus.due_for_clean(clean_version=1) == []


async def test_labels_follow_content_not_cleaner_version(corpus: CorpusStore) -> None:
    await _fetched(corpus)
    await corpus.record_clean(
        URL_A, content_hash="h1", clean_version=1, clean_markdown="## x", word_count=400, error=None
    )
    assert len(await corpus.due_for_label(label_version=1)) == 1

    await corpus.record_labels(
        URL_A,
        labels=ArticleLabels(topic="t"),
        labelled_hash="h1",
        label_version=1,
        labeler_model="m",
    )
    assert await corpus.due_for_label(label_version=1) == []

    # A cleaner-rule bump re-cleans but must NOT re-bill the labelling LLM.
    await corpus.record_clean(
        URL_A, content_hash="h1", clean_version=2, clean_markdown="## y", word_count=400, error=None
    )
    assert await corpus.due_for_label(label_version=1) == []
    assert len(await corpus.due_for_label(label_version=2)) == 1  # prompt changed


async def test_usable_excludes_labels_for_stale_content(corpus: CorpusStore) -> None:
    await _fetched(corpus)
    await corpus.record_clean(
        URL_A, content_hash="h1", clean_version=1, clean_markdown="## x", word_count=400, error=None
    )
    await corpus.record_labels(
        URL_A,
        labels=ArticleLabels(topic="t"),
        labelled_hash="h1",
        label_version=1,
        labeler_model="m",
    )
    assert len(await corpus.usable()) == 1

    # Content changes and is re-cleaned: the old labels describe old text.
    await corpus.record_content(
        URL_A, content_hash="h2", r2_raw_key="k2", title=None, meta_description=None
    )
    await corpus.record_clean(
        URL_A, content_hash="h2", clean_version=1, clean_markdown="## z", word_count=400, error=None
    )
    assert await corpus.usable() == []


async def test_duplicates_flag_is_exact(corpus: CorpusStore) -> None:
    for url in (URL_A, URL_B):
        await _fetched(corpus, url)
        await corpus.record_clean(
            url,
            content_hash="h1",
            clean_version=1,
            clean_markdown="## x",
            word_count=400,
            error=None,
        )
    await corpus.set_duplicates({URL_B})
    await corpus.set_duplicates(set())  # later run: no longer a duplicate
    assert not any(a.is_duplicate for a in await corpus.cleaned())


async def test_runs_are_logged(corpus: CorpusStore) -> None:
    run_id = await corpus.start_run("crawl")
    await corpus.finish_run(run_id, stats={"new": 3})
    [run] = await corpus.recent_runs(1)
    assert run["stage"] == "crawl" and run["status"] == "succeeded" and run["stats"] == {"new": 3}


async def test_labels_roundtrip_as_jsonb(corpus: CorpusStore) -> None:
    await _fetched(corpus)
    await corpus.record_labels(
        URL_A,
        labels=ArticleLabels(topic="t", keywords=["a", "b"], length="short"),
        labelled_hash="h1",
        label_version=1,
        labeler_model="m",
    )
    row = await corpus.get(URL_A)
    assert row is not None and row.labels is not None
    assert row.labels.keywords == ["a", "b"] and row.labels.length == "short"


def test_every_public_table_has_rls_enabled(pg_dsn: str) -> None:
    """Deny-by-default must hold for every table, including the runner's own."""
    import psycopg

    with psycopg.connect(pg_dsn) as conn:
        rows = conn.execute(
            "select relname from pg_class c join pg_namespace n on n.oid = c.relnamespace "
            "where n.nspname = 'public' and c.relkind = 'r' and not c.relrowsecurity"
        ).fetchall()
    assert rows == [], f"tables without RLS: {[r[0] for r in rows]}"


# ------------------------------------------------------------ pure (no database)


def test_split_is_deterministic() -> None:
    url = "https://www.jenosize.com/en/ideas/futurist/example"
    assert split_for(url, 0.1) == split_for(url, 0.1)


def test_split_respects_the_requested_fraction() -> None:
    urls = [f"https://example.com/{i}" for i in range(2000)]
    held_out = sum(1 for u in urls if split_for(u, 0.1) == "eval")
    assert 0.07 < held_out / len(urls) < 0.13


@pytest.mark.parametrize(("frac", "expected"), [(0.0, "train"), (1.0, "eval")])
def test_split_extremes(frac: float, expected: str) -> None:
    assert split_for("https://x", frac) == expected
