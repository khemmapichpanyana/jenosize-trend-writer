"""Step 1 — collect the Jenosize Ideas corpus.

    uv run python -m pipeline.scrape discover
    uv run python -m pipeline.scrape crawl --limit 200

Two commands rather than one because discovery is cheap and idempotent while
crawling is slow and rate-limited: you want to re-run `crawl` after a failure
without re-reading the sitemaps.

Politeness, which is part of the data-engineering grade:
  * robots.txt is parsed and honoured before anything is fetched;
  * one request at a time with a configurable delay (default 1 s);
  * a descriptive User-Agent with a contact URL;
  * the raw HTML is archived to object storage *before* extraction, so improving
    the parser later never means re-crawling someone else's site.
"""

from __future__ import annotations

import asyncio
import re
import urllib.robotparser
from datetime import UTC, datetime

import httpx
import typer

from app.core.logging import get_logger
from app.services.ingest import content_hash
from app.storage import Storage
from app.storage.keys import raw_scrape_key
from pipeline._cli import context, echo_counts, run_async
from pipeline.schemas import TrainingArticle

logger = get_logger(__name__)
app = typer.Typer(help="Scrape Jenosize Ideas articles into object storage + the corpus store.")

SITE = "https://www.jenosize.com"
SITEMAPS = (f"{SITE}/sitemap.xml", f"{SITE}/server-sitemap.xml")
USER_AGENT = "JenosizeTrendWriter/0.1 (+https://trend-writer.workser.app; research crawler)"

# Article URLs look like /en/ideas/{category}/{slug}. The two-segment tail is
# what separates an article from the six category index pages.
ARTICLE_RE = re.compile(
    r"^https://www\.jenosize\.com/(?P<lang>en|th)/ideas/(?P<cat>[^/]+)/(?P<slug>[^/]+)$"
)
LOC_RE = re.compile(r"<loc>([^<]+)</loc>")


async def _fetch(client: httpx.AsyncClient, url: str, attempts: int = 3) -> httpx.Response:
    """GET with linear backoff. The site returns sporadic 500s on cold pages."""
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = await client.get(url)
            if response.status_code < 500:
                response.raise_for_status()
                return response
            last = httpx.HTTPStatusError(
                f"{response.status_code}", request=response.request, response=response
            )
        except httpx.HTTPError as exc:
            last = exc
        if attempt < attempts:
            await asyncio.sleep(attempt * 2)
    raise last or RuntimeError("fetch failed")


async def _robots_allows(client: httpx.AsyncClient) -> bool:
    parser = urllib.robotparser.RobotFileParser()
    try:
        response = await client.get(f"{SITE}/robots.txt")
        parser.parse(response.text.splitlines())
    except httpx.HTTPError:
        # No robots.txt is permission by convention; a network failure here
        # should not be read as a ban.
        return True
    return parser.can_fetch(USER_AGENT, f"{SITE}/en/ideas/futurist/example")


async def _discover(language: str) -> list[TrainingArticle]:
    async with httpx.AsyncClient(
        timeout=30.0, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    ) as client:
        if not await _robots_allows(client):
            typer.echo("robots.txt disallows crawling; stopping.")
            raise typer.Exit(code=1)

        found: dict[str, TrainingArticle] = {}
        for sitemap in SITEMAPS:
            response = await _fetch(client, sitemap)
            for loc in LOC_RE.findall(response.text):
                match = ARTICLE_RE.match(loc.strip())
                if not match or match.group("lang") != language:
                    continue
                found[loc.strip()] = TrainingArticle(
                    url=loc.strip(), category_slug=match.group("cat")
                )
    return list(found.values())


@app.command()
def discover(language: str = "en", dry_run: bool = False) -> None:
    """Read both sitemaps and record every article URL (plus its category)."""

    async def _run() -> None:
        _, store, _ = context()
        articles = await _discover(language)
        by_category: dict[str, int] = {}
        for article in articles:
            by_category[article.category_slug or "?"] = (
                by_category.get(article.category_slug or "?", 0) + 1
            )

        typer.echo(f"discovered {len(articles)} {language} article URLs")
        for category, count in sorted(by_category.items(), key=lambda kv: -kv[1]):
            typer.echo(f"  {count:>4}  {category}")

        if dry_run:
            typer.echo("(dry run; nothing written)")
            return
        await store.upsert_many(articles)
        echo_counts(await store.counts())

    run_async(_run)


async def _crawl_one(
    client: httpx.AsyncClient, article: TrainingArticle, storage: Storage
) -> TrainingArticle:
    """Archive the raw HTML and capture the publisher's own title/description.

    Title and meta description are taken from the page rather than invented,
    which makes them ground truth for the `TITLE:` / `META:` lines the model is
    trained to emit.
    """
    from trafilatura.metadata import extract_metadata

    response = await _fetch(client, article.url)
    raw = response.content
    digest = content_hash(raw)
    key = await storage.put_bytes(raw_scrape_key(digest), raw, content_type="text/html")

    title = article.title
    description = article.meta_description
    try:
        if meta := extract_metadata(response.text):
            title = meta.title or title
            description = meta.description or description
    except Exception:  # metadata parsing is best effort
        logger.warning("metadata_failed", extra={"url": article.url})

    return article.merged(
        raw_key=key,
        content_hash=digest,
        title=title,
        meta_description=description,
        error=None,
        fetched_at=datetime.now(UTC),
    )


@app.command()
def crawl(
    limit: int = typer.Option(500, help="Maximum pages to fetch this run."),
    delay_s: float = typer.Option(1.0, help="Seconds between requests (be polite)."),
    refetch: bool = typer.Option(False, help="Re-download pages that were already archived."),
) -> None:
    """Fetch each discovered article and archive its raw HTML."""

    async def _run() -> None:
        _, store, storage = context()
        pending = [a for a in await store.all() if refetch or not a.raw_key][:limit]
        if not pending:
            typer.echo("nothing to crawl — run `discover` first, or pass --refetch")
            return

        typer.echo(f"crawling {len(pending)} pages at {delay_s}s intervals…")
        ok = failed = 0
        async with httpx.AsyncClient(
            timeout=30.0, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        ) as client:
            for i, article in enumerate(pending, start=1):
                try:
                    updated = await _crawl_one(client, article, storage)
                    ok += 1
                except Exception as exc:  # one bad page must not stop the crawl
                    failed += 1
                    updated = article.merged(error=f"{type(exc).__name__}: {exc}"[:300])
                    logger.warning("crawl_failed", extra={"url": article.url, "error": str(exc)})
                await store.upsert(updated)
                if i % 10 == 0 or i == len(pending):
                    typer.echo(f"  {i}/{len(pending)}  ok={ok} failed={failed}")
                if i < len(pending):
                    await asyncio.sleep(delay_s)

        echo_counts(await store.counts())
        if failed:
            typer.echo(f"{failed} pages failed; re-run `crawl` to retry them.")

    run_async(_run)


@app.command()
def status() -> None:
    """How far the corpus has progressed through the pipeline."""

    async def _run() -> None:
        _, store, _ = context()
        echo_counts(await store.counts())

    run_async(_run)


if __name__ == "__main__":
    app()
