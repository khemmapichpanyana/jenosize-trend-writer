"""Step 2 — turn raw HTML into training-grade markdown.

    uv run python -m pipeline.clean run --min-words 300

STATUS: stub.

The cleaning rules decide what the model learns to imitate, so they are the
highest-leverage part of the data pipeline:
  * drop boilerplate (nav, cookie banners, "related articles", CTAs);
  * drop articles under ~300 words — they teach the model to stop early;
  * normalise headings to `##` so the fine-tune sees one consistent structure;
  * strip author bylines and dates, which the model must never invent;
  * deduplicate near-identical articles (shingle/MinHash) — repeats overfit.
"""

from __future__ import annotations

import typer

app = typer.Typer(help="Clean scraped HTML into training markdown.")


@app.command()
def run(min_words: int = 300, max_words: int = 4000) -> None:
    """Clean every `training_articles` row that has raw HTML but no markdown."""
    # TODO(day-1): html -> markdown, apply the filters above, write clean_markdown.
    typer.echo(f"TODO: clean articles, keeping {min_words}-{max_words} words")


@app.command()
def stats() -> None:
    """Corpus health: count, word-count distribution, duplicate rate."""
    # TODO(day-1): feeds docs/data_card.md.
    typer.echo("TODO: corpus statistics")


if __name__ == "__main__":
    app()
