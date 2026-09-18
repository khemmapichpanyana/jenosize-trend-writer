"""End to end: re-running the pipeline writes only what is genuinely new.

A fake jenosize.com (respx) serves real-shaped pages whose raw HTML changes on
every request — exactly like the live site, where Cloudflare rewrites its
email-protection tokens per response. The pipeline runs against a real Postgres
and an S3-compatible R2 stand-in, twice, and must not write anything the second
time. Then one article is edited, and exactly that one must flow through.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import httpx
import pytest
import respx

from pipeline import build_dataset, clean, label, scrape
from pipeline.store import CorpusStore
from tests.conftest import r2_keys

SITE = "https://www.jenosize.com"
SLUGS = ["agentic-ai", "retail-media", "green-logistics"]


def _article_html(slug: str, edition: int = 1) -> str:
    """A page shaped like the real site: <h4> sections, CTA, reference list, and
    a Cloudflare token that is different on every single response."""
    token = uuid.uuid4().hex
    sections = "".join(
        f"<h4>Section {i} on {slug}</h4>"
        + "".join(
            f"<p>Paragraph {j} of section {i} explains how {slug.replace('-', ' ')} changes "
            f"operating models for regional enterprises, edition {edition}, with concrete "
            f"examples drawn from markets across Southeast Asia and practical next steps.</p>"
            for j in range(4)
        )
        for i in range(4)
    )
    return f"""<html><head><title>{slug.title()} Title</title>
<meta name="description" content="Meta for {slug}."></head><body>
<nav>Home Ideas Contact</nav>
<article><h4>{slug.title()} Title</h4>{sections}
<h4>Call to Action</h4><p>Contact us today via our website.</p>
<h5>References</h5><p>Harvard Business Review, a study.</p></article>
<footer><a href="/cdn-cgi/l/email-protection#{token}">Email</a></footer>
</body></html>"""


class FakeSite:
    def __init__(self) -> None:
        self.editions = dict.fromkeys(SLUGS, 1)
        self.requests = 0
        self.truncate_next: dict[str, int] = {}  # slug -> how many truncated replies

    def install(self, router: respx.Router) -> None:
        router.get(f"{SITE}/robots.txt").respond(200, text="User-agent: *\nAllow: /\n")
        locs = "".join(f"<url><loc>{SITE}/en/ideas/futurist/{s}</loc></url>" for s in SLUGS)
        router.get(f"{SITE}/sitemap.xml").respond(200, text=f"<urlset>{locs}</urlset>")
        router.get(f"{SITE}/server-sitemap.xml").respond(
            200, text=f"<urlset><url><loc>{SITE}/en/ideas/futurist</loc></url></urlset>"
        )
        router.get(url__regex=rf"{SITE}/en/ideas/futurist/[a-z-]+$").mock(side_effect=self._page)

    def _page(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        slug = request.url.path.rsplit("/", 1)[-1]
        html = _article_html(slug, self.editions[slug])
        if self.truncate_next.get(slug):
            # What the live site did once: HTTP 200, body cut off mid-article.
            self.truncate_next[slug] -= 1
            html = html[: len(html) // 2]
        return httpx.Response(200, text=html)


class FakeLabeler:
    """Duck-types `AsyncOpenAI.chat.completions.create` and counts calls."""

    def __init__(self) -> None:
        self.calls = 0
        self.chat = self
        self.completions = self

    async def create(self, **_: Any) -> Any:
        self.calls += 1
        reply = {"topic": f"Topic {self.calls}", "industry": "tech", "keywords": ["ai", "retail"]}
        message = type("M", (), {"content": json.dumps(reply)})
        choice = type("C", (), {"message": message})
        return type("R", (), {"choices": [choice]})


@pytest.fixture
def site() -> Any:
    fake = FakeSite()
    with respx.mock(assert_all_called=False) as router:
        fake.install(router)
        yield fake


async def _full_run(corpus: CorpusStore, r2: Any, labeler: FakeLabeler, *, recheck: bool) -> dict:
    async with scrape.http_client() as client:
        discovered = await scrape.run_discover(corpus, client)
        crawled = await scrape.run_crawl(
            corpus, r2, client, delay_s=0, recheck_days=0 if recheck else None
        )
    cleaned = await clean.run_clean(corpus, r2)
    labelled, _ = await label.run_label(corpus, labeler, "fake-model")  # type: ignore[arg-type]
    return {"discover": discovered, "crawl": crawled, "clean": cleaned, "label": labelled}


async def test_second_run_writes_nothing(corpus: CorpusStore, r2: Any, site: FakeSite) -> None:
    labeler = FakeLabeler()

    first = await _full_run(corpus, r2, labeler, recheck=False)
    assert first["discover"]["new"] == 3
    assert first["crawl"]["new"] == 3
    assert first["clean"]["cleaned"] == 3
    assert first["label"]["labelled"] == 3
    objects_after_first = r2_keys(r2)
    assert len(objects_after_first) == 3  # one raw HTML archive per article

    # Second run, re-fetching every page. Raw HTML differs byte-for-byte (fresh
    # Cloudflare token), but the content did not change, so nothing is written.
    second = await _full_run(corpus, r2, labeler, recheck=True)
    assert second["discover"]["new"] == 0
    assert second["crawl"] == {**second["crawl"], "unchanged": 3, "new": 0, "changed": 0}
    assert second["clean"]["pending"] == 0
    assert second["label"]["pending"] == 0
    assert r2_keys(r2) == objects_after_first
    assert labeler.calls == 3  # no LLM spend on unchanged content


async def test_without_recheck_known_pages_are_not_even_fetched(
    corpus: CorpusStore, r2: Any, site: FakeSite
) -> None:
    await _full_run(corpus, r2, FakeLabeler(), recheck=False)
    fetched = site.requests
    await _full_run(corpus, r2, FakeLabeler(), recheck=False)
    assert site.requests == fetched  # zero page requests on a no-news run


async def test_an_edited_article_flows_through_alone(
    corpus: CorpusStore, r2: Any, site: FakeSite
) -> None:
    labeler = FakeLabeler()
    await _full_run(corpus, r2, labeler, recheck=False)

    site.editions["retail-media"] = 2  # the publisher edits one article
    run = await _full_run(corpus, r2, labeler, recheck=True)

    assert run["crawl"]["changed"] == 1 and run["crawl"]["unchanged"] == 2
    assert run["clean"]["pending"] == 1
    assert run["label"]["pending"] == 1
    assert len(r2_keys(r2)) == 4  # the new version is archived; the old one is kept
    assert labeler.calls == 4


async def test_cleaning_rules_reach_real_shaped_pages(
    corpus: CorpusStore, r2: Any, site: FakeSite
) -> None:
    await _full_run(corpus, r2, FakeLabeler(), recheck=False)
    [article, *_] = await corpus.cleaned()
    md = article.clean_markdown or ""
    assert "## Section 0" in md  # <h4> became ##
    assert "Call to Action" not in md and "Contact us" not in md
    assert "Harvard" not in md  # reference list dropped
    assert not md.startswith("## Agentic-Ai Title")  # restated title removed


async def test_dataset_versions_are_published_once_and_immutable(
    corpus: CorpusStore, r2: Any, site: FakeSite
) -> None:
    labeler = FakeLabeler()
    await _full_run(corpus, r2, labeler, recheck=False)

    first = await build_dataset.run_build(corpus, r2, version="v1", eval_frac=0.0)
    assert first["status"] == "published" and first["train"] == 3
    assert "datasets/v1/train.jsonl" in r2_keys(r2)

    again = await build_dataset.run_build(corpus, r2, version="v1", eval_frac=0.0)
    assert again["status"] == "unchanged"

    site.editions["agentic-ai"] = 2
    await _full_run(corpus, r2, labeler, recheck=True)
    with pytest.raises(build_dataset.DatasetError, match="immutable"):
        await build_dataset.run_build(corpus, r2, version="v1", eval_frac=0.0)

    v2 = await build_dataset.run_build(corpus, r2, version="v2", eval_frac=0.0)
    assert v2["status"] == "published"
    assert await corpus.dataset_version("v1") is not None  # v1 still intact


async def test_published_dataset_passes_validation_from_r2(
    corpus: CorpusStore, r2: Any, site: FakeSite
) -> None:
    await _full_run(corpus, r2, FakeLabeler(), recheck=False)
    await build_dataset.run_build(corpus, r2, version="v1", eval_frac=0.0)

    rows = [
        json.loads(line)
        for line in (await r2.get_bytes("datasets/v1/train.jsonl")).decode().splitlines()
    ]
    assert build_dataset.validate_rows({"train": rows, "eval": []}) == []


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_: float) -> None:
        return None

    monkeypatch.setattr(scrape.asyncio, "sleep", instant)


async def test_a_truncated_response_is_retried_not_stored(
    corpus: CorpusStore, r2: Any, site: FakeSite, no_backoff: None
) -> None:
    labeler = FakeLabeler()
    await _full_run(corpus, r2, labeler, recheck=False)
    objects = r2_keys(r2)

    site.truncate_next["retail-media"] = 1  # one bad reply, then a good one
    run = await _full_run(corpus, r2, labeler, recheck=True)
    assert run["crawl"]["unchanged"] == 3 and run["crawl"]["changed"] == 0
    assert r2_keys(r2) == objects


async def test_a_persistently_truncated_page_keeps_its_good_version(
    corpus: CorpusStore, r2: Any, site: FakeSite, no_backoff: None
) -> None:
    await _full_run(corpus, r2, FakeLabeler(), recheck=False)
    good = await corpus.get(f"{SITE}/en/ideas/futurist/retail-media")

    site.truncate_next["retail-media"] = 99
    run = await _full_run(corpus, r2, FakeLabeler(), recheck=True)
    after = await corpus.get(f"{SITE}/en/ideas/futurist/retail-media")

    assert run["crawl"]["failed"] == 1
    assert good is not None and after is not None
    assert after.content_hash == good.content_hash  # good content never overwritten
    assert after.clean_markdown == good.clean_markdown
