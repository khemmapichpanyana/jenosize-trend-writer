"""Step 1 — collect the Jenosize Ideas corpus.

    uv run python -m pipeline.scrape crawl --limit 200

STATUS: stub. The CLI shape, storage keys and DB table are final; the crawl
itself is Day 1.

Ethics/robustness notes that matter for the data-engineering grade:
  * respect robots.txt and rate-limit to ~1 request/second;
  * archive the raw HTML to R2 before extracting anything, so a better parser
    later does not mean re-crawling someone else's site;
  * dedupe on sha256 of the raw bytes (`source_documents.content_hash`).
"""

from __future__ import annotations

import typer

app = typer.Typer(help="Scrape Jenosize Ideas articles into R2 + Postgres.")

SITEMAP_URL = "https://www.jenosize.com/sitemap.xml"


@app.command()
def discover(sitemap: str = SITEMAP_URL, limit: int = 500) -> None:
    """List candidate article URLs from the sitemap."""
    # TODO(day-1): fetch the sitemap, filter to /en/ideas/ paths, print URLs.
    typer.echo(f"TODO: discover up to {limit} article URLs from {sitemap}")


@app.command()
def crawl(limit: int = 200, delay_s: float = 1.0) -> None:
    """Fetch each article, archive the raw HTML, upsert `training_articles`."""
    # TODO(day-1): for each URL -> httpx GET -> raw/scrape/{date}/{hash}.html ->
    # app.services.ingest.extract_html -> training_articles row.
    typer.echo(f"TODO: crawl {limit} articles at {delay_s}s intervals")


if __name__ == "__main__":
    app()
