"""Step 1 — collect the Jenosize Ideas corpus into Postgres + R2, incrementally.

    uv run python -m pipeline.scrape discover     # sitemap -> new URLs only
    uv run python -m pipeline.scrape crawl        # fetch pages we don't have yet
    uv run python -m pipeline.scrape crawl --recheck-days 30   # also re-check old ones
    uv run python -m pipeline.scrape status

Safe to run on a schedule: a run with nothing new writes nothing.

How "new" is decided — measured, not assumed:
  * Sitemap <lastmod> is useless here: it is the site's build time, identical for
    every article.
  * Raw HTML is useless too: Cloudflare rewrites email-protection tokens on every
    response, so two fetches of an unchanged page never match byte-for-byte, and
    the site sends no ETag / Last-Modified.
  * The *extracted article text* is stable. Its sha256 is the fingerprint. Raw
    HTML is uploaded to R2 only when the fingerprint is new or has changed.

Politeness: robots.txt honoured, one request at a time with a delay, a
descriptive User-Agent, and raw HTML archived so re-parsing never re-crawls.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import urllib.robotparser
from typing import Any

import httpx
import typer

from app.core.logging import get_logger
from app.storage.base import Storage
from app.storage.keys import raw_scrape_key
from pipeline._cli import echo_stats, pipeline_context, recorded_run, run_async
from pipeline.clean import CLEAN_VERSION
from pipeline.label import LABEL_VERSION
from pipeline.schemas import TrainingArticle
from pipeline.store import CorpusStore

logger = get_logger(__name__)
app = typer.Typer(help="Scrape Jenosize Ideas into Postgres + R2 (incremental).")

SITE = "https://www.jenosize.com"
SITEMAPS = (f"{SITE}/sitemap.xml", f"{SITE}/server-sitemap.xml")
USER_AGENT = "JenosizeTrendWriter/0.1 (+https://trend-writer.workser.app; research crawler)"

# /en/ideas/{category}/{slug}. The two-segment tail separates an article from the
# six category index pages.
ARTICLE_RE = re.compile(
    r"^https://www\.jenosize\.com/(?P<lang>en|th)/ideas/(?P<cat>[^/]+)/(?P<slug>[^/]+)$"
)
LOC_RE = re.compile(r"<loc>([^<]+)</loc>")
_WS = re.compile(r"\s+")


def http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=30.0, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    )


def fingerprint(html: str, url: str | None = None) -> str | None:
    """sha256 of the article's extracted text, or None if there is no article.

    Whitespace is collapsed first so a template reflow cannot register as an
    edit. See the module docstring for why raw HTML cannot be the fingerprint.
    """
    import trafilatura

    text = trafilatura.extract(html, url=url, favor_precision=True, include_comments=False) or ""
    text = _WS.sub(" ", text).strip()
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def page_metadata(html: str) -> tuple[str | None, str | None]:
    """The publisher's own <title> and meta description: real training targets."""
    from trafilatura.metadata import extract_metadata

    try:
        meta = extract_metadata(html)
    except Exception:  # metadata is best effort
        return None, None
    return (meta.title, meta.description) if meta else (None, None)


class TruncatedResponseError(httpx.HTTPError):
    """A 200 whose body stopped early. Retryable; must never be stored."""


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    attempts: int = 3,
    *,
    complete_marker: str | None = None,
) -> httpx.Response:
    """GET with linear backoff; the site returns sporadic 5xx on cold pages.

    `complete_marker` guards against truncated bodies. Observed live: a re-fetch
    of an unchanged article returned HTTP 200 with half the HTML (no closing
    `</html>`, last sections missing). Its fingerprint differed, so it was
    recorded as an "edit" and replaced the complete version. A body that does not
    end with the marker is now treated like a 5xx: retried, never stored.
    """
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = await client.get(url)
            if response.status_code < 500:
                response.raise_for_status()
                if complete_marker and complete_marker not in response.text[-4096:].lower():
                    raise TruncatedResponseError(
                        f"truncated body ({len(response.content)} bytes, no {complete_marker})"
                    )
                return response
            last = httpx.HTTPStatusError(
                f"HTTP {response.status_code}", request=response.request, response=response
            )
        except httpx.HTTPStatusError:
            raise  # a 4xx will not fix itself on retry
        except httpx.HTTPError as exc:
            last = exc
        if attempt < attempts:
            await asyncio.sleep(attempt * 2)
    raise last or RuntimeError("fetch failed")


async def robots_allows(client: httpx.AsyncClient) -> bool:
    parser = urllib.robotparser.RobotFileParser()
    try:
        response = await client.get(f"{SITE}/robots.txt")
        parser.parse(response.text.splitlines())
    except httpx.HTTPError:
        return True  # no robots.txt is permission by convention
    return parser.can_fetch(USER_AGENT, f"{SITE}/en/ideas/futurist/example")


async def discover_articles(
    client: httpx.AsyncClient, language: str = "en"
) -> list[TrainingArticle]:
    found: dict[str, TrainingArticle] = {}
    for sitemap in SITEMAPS:
        response = await fetch(client, sitemap)
        for loc in LOC_RE.findall(response.text):
            url = loc.strip()
            match = ARTICLE_RE.match(url)
            if match and match.group("lang") == language:
                found[url] = TrainingArticle(url=url, category_slug=match.group("cat"))
    return list(found.values())


async def run_discover(
    store: CorpusStore, client: httpx.AsyncClient, *, language: str = "en"
) -> dict[str, Any]:
    if not await robots_allows(client):
        raise RuntimeError("robots.txt disallows crawling")
    articles = await discover_articles(client, language)
    new = await store.add_urls(articles)
    return {"in_sitemap": len(articles), "new": len(new), "known": len(articles) - len(new)}


async def crawl_one(
    store: CorpusStore, storage: Storage, client: httpx.AsyncClient, article: TrainingArticle
) -> str:
    """Fetch one page; write to R2/Postgres only if its content is new or changed.

    Returns "new", "changed", "unchanged" or "empty".
    """
    response = await fetch(client, article.url, complete_marker="</html>")
    html = response.text
    content_hash = fingerprint(html, article.url)
    if content_hash is None:
        await store.record_error(article.url, "crawl: no article text on page", checked=True)
        return "empty"

    previous = await store.current_hash(article.url)
    if previous == content_hash:
        await store.mark_unchanged(article.url)
        return "unchanged"

    title, description = page_metadata(html)
    key = raw_scrape_key(content_hash)
    # Content-addressed key: if this exact version is already archived (e.g. a
    # crash between the upload and the DB write), skip the upload.
    if not await storage.exists(key):
        await storage.put_bytes(key, response.content, content_type="text/html; charset=utf-8")
    await store.record_content(
        article.url,
        content_hash=content_hash,
        r2_raw_key=key,
        title=title,
        meta_description=description,
    )
    return "new" if previous is None else "changed"


async def run_crawl(
    store: CorpusStore,
    storage: Storage,
    client: httpx.AsyncClient,
    *,
    limit: int = 1000,
    delay_s: float = 1.0,
    recheck_days: float | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    if not await robots_allows(client):
        raise RuntimeError("robots.txt disallows crawling")

    pending = await store.due_for_crawl(limit=limit, recheck_days=recheck_days)
    stats = {"checked": 0, "new": 0, "changed": 0, "unchanged": 0, "empty": 0, "failed": 0}
    for i, article in enumerate(pending, start=1):
        try:
            outcome = await crawl_one(store, storage, client, article)
            stats[outcome] += 1
        except Exception as exc:  # one bad page must not stop the crawl
            stats["failed"] += 1
            permanent = isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500
            await store.record_error(article.url, f"crawl: {exc}", checked=permanent)
            logger.warning("crawl_failed", extra={"url": article.url, "error": str(exc)})
        stats["checked"] += 1
        if progress and (i % 10 == 0 or i == len(pending)):
            typer.echo(
                f"  {i}/{len(pending)}  "
                + "  ".join(f"{k}={v}" for k, v in stats.items() if k != "checked")
            )
        if i < len(pending) and delay_s:
            await asyncio.sleep(delay_s)
    return stats


# ---------------------------------------------------------------------------- CLI


@app.command()
def discover(language: str = "en") -> None:
    """Read the sitemaps and record URLs we have never seen."""

    async def _run() -> None:
        async with pipeline_context() as (_, store, _), http_client() as client:
            async with recorded_run(store, "discover") as stats:
                stats.update(await run_discover(store, client, language=language))
            echo_stats("discover", stats)

    run_async(_run)


@app.command()
def crawl(
    limit: int = typer.Option(1000, help="Maximum pages to fetch this run."),
    delay_s: float = typer.Option(1.0, help="Seconds between requests (be polite)."),
    recheck_days: float | None = typer.Option(
        None, help="Also re-fetch pages last checked more than N days ago, to catch edits."
    ),
) -> None:
    """Fetch pages that are new (or stale, with --recheck-days); archive only real changes."""

    async def _run() -> None:
        async with pipeline_context() as (_, store, storage), http_client() as client:
            async with recorded_run(store, "crawl") as stats:
                stats.update(
                    await run_crawl(
                        store,
                        storage,
                        client,
                        limit=limit,
                        delay_s=delay_s,
                        recheck_days=recheck_days,
                        progress=True,
                    )
                )
            echo_stats("crawl", stats)
            if stats.get("failed"):
                typer.echo("failed pages are retried automatically on the next `crawl`.")

    run_async(_run)


@app.command()
def status(runs: int = typer.Option(8, help="How many recent runs to show.")) -> None:
    """Corpus progress per stage, and the most recent pipeline runs."""

    async def _run() -> None:
        async with pipeline_context() as (_, store, _):
            counts = await store.counts(clean_version=CLEAN_VERSION, label_version=LABEL_VERSION)
            echo_stats("corpus", counts)
            for run in await store.recent_runs(runs):
                when = run["started_at"].strftime("%Y-%m-%d %H:%M")
                detail = run["error"] or "  ".join(
                    f"{k}={v}" for k, v in (run["stats"] or {}).items()
                )
                typer.echo(f"  {when}  {run['stage']:<8} {run['status']:<9} {detail}")

    run_async(_run)


if __name__ == "__main__":
    app()
