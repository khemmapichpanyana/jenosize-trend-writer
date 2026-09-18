"""Corpus store: the pipeline's resumability depends on these semantics."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.schemas import ArticleLabels, TrainingArticle
from pipeline.store import SqliteCorpusStore, split_for


@pytest.fixture
def store(tmp_path: Path) -> SqliteCorpusStore:
    return SqliteCorpusStore(str(tmp_path / "corpus.db"))


async def test_roundtrip_preserves_labels(store: SqliteCorpusStore) -> None:
    article = TrainingArticle(
        url="https://example.com/a",
        category_slug="futurist",
        title="A Title",
        clean_markdown="## Body",
        word_count=2,
        labels=ArticleLabels(topic="a topic", keywords=["one", "two"], length="short"),
    )
    await store.upsert(article)

    loaded = await store.get("https://example.com/a")
    assert loaded is not None
    assert loaded.labels is not None
    assert loaded.labels.keywords == ["one", "two"]
    assert loaded.labels.length == "short"
    assert loaded.category_slug == "futurist"


async def test_later_stages_do_not_wipe_earlier_output(store: SqliteCorpusStore) -> None:
    # This is what makes `clean` re-runnable after `scrape`: an upsert carrying
    # only the new columns must leave the old ones intact.
    await store.upsert(
        TrainingArticle(url="https://example.com/a", title="Scraped", raw_key="raw/a.html")
    )
    await store.upsert(TrainingArticle(url="https://example.com/a", clean_markdown="## Cleaned"))

    loaded = await store.get("https://example.com/a")
    assert loaded is not None
    assert loaded.raw_key == "raw/a.html"
    assert loaded.title == "Scraped"
    assert loaded.clean_markdown == "## Cleaned"


async def test_counts_track_pipeline_progress(store: SqliteCorpusStore) -> None:
    await store.upsert_many(
        [
            TrainingArticle(url="https://a", raw_key="k1"),
            TrainingArticle(url="https://b", raw_key="k2", clean_markdown="## x"),
            TrainingArticle(url="https://c", error="boom"),
        ]
    )
    counts = await store.counts()
    assert counts["discovered"] == 3
    assert counts["scraped"] == 2
    assert counts["cleaned"] == 1
    assert counts["errors"] == 1


async def test_upsert_many_is_a_noop_on_empty(store: SqliteCorpusStore) -> None:
    assert await store.upsert_many([]) == 0


def test_split_is_deterministic_and_stable() -> None:
    url = "https://www.jenosize.com/en/ideas/futurist/example"
    assert split_for(url, 0.1) == split_for(url, 0.1)
    # Adding articles later must not move an existing one across the split,
    # which is why the hash is of the URL alone.
    assert split_for(url, 0.1, seed=13) == split_for(url, 0.1, seed=13)


def test_split_respects_the_requested_fraction() -> None:
    urls = [f"https://example.com/{i}" for i in range(2000)]
    held_out = sum(1 for u in urls if split_for(u, 0.1) == "eval")
    assert 0.07 < held_out / len(urls) < 0.13


def test_split_extremes() -> None:
    assert split_for("https://x", 0.0) == "train"
    assert split_for("https://x", 1.0) == "eval"
