"""Step 3 — reverse-label each article with the brief that would have produced it.

    uv run python -m pipeline.label run --limit 50

STATUS: stub.

This is the core data-engineering trick of the project. We have articles but no
briefs, and the service is trained on (brief -> article). So we run a strong
OpenAI-compatible model *backwards*: given the finished article, infer the
topic, category, industry, audience and SEO keywords that it answers.

Two rules keep the labels honest:
  * the labeller only ever sees the article, never a human-written brief, so it
    cannot leak the answer;
  * every inferred industry goes through `app.services.normalize.normalize_industry`,
    so training labels use exactly the vocabulary the API produces at inference.
"""

from __future__ import annotations

import typer

app = typer.Typer(help="Reverse-label cleaned articles into training briefs.")

LABEL_SCHEMA = {
    "topic": "str",
    "category": "str",
    "industry": "str (canonical list)",
    "audience": "str",
    "keywords": "list[str], 3-8 items",
    "length": "short|medium|long, derived from word_count",
}


@app.command()
def run(limit: int = 100, model: str | None = None) -> None:
    """Infer a brief per article and store it in `training_articles.labels`."""
    # TODO(day-1): call LABELER_BASE_URL with a JSON-schema-constrained prompt,
    # validate against LABEL_SCHEMA, normalise, then persist.
    typer.echo(f"TODO: label {limit} articles with {model or 'LABELER_MODEL'}")


@app.command()
def audit(sample: int = 20) -> None:
    """Print a random sample of (article, inferred brief) pairs for human review."""
    # TODO(day-1): label quality is checked by reading, not by a metric.
    typer.echo(f"TODO: audit {sample} labelled pairs")


if __name__ == "__main__":
    app()
