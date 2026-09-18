"""Step 3 — reverse-label each article with the brief that would have produced it.

    uv run python -m pipeline.label run --limit 200
    uv run python -m pipeline.label audit --sample 5

The problem: we have finished articles but no briefs, and the service is trained
on (brief -> article). The fix is to run a strong instruction model *backwards* —
show it the article, ask what brief it answers.

Three rules keep the labels honest:

1. The labeller only ever sees the article body, never a human brief, so it
   cannot copy an answer that does not exist.
2. `category` and `language` are taken from the URL, not inferred. The publisher
   already assigned them; an LLM guess would only add noise.
3. `length` is computed from the real word count. Asking a model to judge length
   when we can count it is strictly worse.

Everything the model *does* infer (topic, industry, audience, keywords) is then
pushed through the very normalizers the API applies at inference time, so the
training labels use exactly the vocabulary production produces.
"""

from __future__ import annotations

import asyncio
import json
import re

import typer
from openai import AsyncOpenAI

from app.core.logging import get_logger
from app.services.normalize import normalize_industry, normalize_keywords, normalize_optional_text
from pipeline._cli import echo_stats, pipeline_context, recorded_run, run_async
from pipeline.clean import detect_language
from pipeline.schemas import ArticleLabels, TrainingArticle
from pipeline.store import CorpusStore

logger = get_logger(__name__)
app = typer.Typer(help="Reverse-label cleaned articles into training briefs.")

# Map the URL's category slug to the label the API exposes. Doing this here
# means `category` in training data and `category` in a user request are the
# same strings.
CATEGORY_NAMES = {
    "futurist": "Futurist",
    "real-time-marketing": "Real-time Marketing",
    "transformation-and-technology": "Transformation and Technology",
    "understand-people-and-consumer": "Understand People and Consumer",
    "experience-the-new-world": "Experience the New World",
    "utility-for-our-world": "Utility for Our World",
}

# Length buckets must match app.schemas.articles.TARGET_WORDS (600/1000/1500).
LENGTH_BOUNDS = ((800, "short"), (1250, "medium"))

LABELER_SYSTEM = """You reverse-engineer content briefs.

Given a finished business article, infer the brief a marketing lead would have
written to commission it. Report only what the article itself supports; never
invent a client, a company or a statistic.

Reply with a single JSON object and nothing else:

{
  "topic": "one specific sentence fragment naming the subject, 5-15 words",
  "industry": "the single industry this most serves, or null if general",
  "audience": "who the article addresses, e.g. 'Marketing leaders'",
  "keywords": ["3-8 lowercase SEO phrases that actually appear in the article"]
}"""

# Bump when LABELER_SYSTEM or build_labels changes meaning: every article is
# then re-labelled on the next run. Labels are otherwise keyed on content only,
# so a cleaning-rule tweak never re-bills the labelling LLM.
LABEL_VERSION = 1

# Enough article to characterise the brief without paying for the whole body.
MAX_ARTICLE_CHARS = 6000
CONCURRENCY = 4

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def length_for(word_count: int) -> str:
    for bound, label in LENGTH_BOUNDS:
        if word_count < bound:
            return label
    return "long"


def language_for(url: str) -> str:
    return "th" if "/th/" in url else "en"


def parse_label_json(raw: str) -> dict:
    """Extract the JSON object from a model reply.

    Small models wrap JSON in prose or a ```json fence even when told not to, and
    a whole labelling run should not fail on a stray backtick.
    """
    match = _JSON_BLOCK.search(raw.strip())
    if not match:
        raise ValueError(f"no JSON object in reply: {raw[:200]!r}")
    return json.loads(match.group(0))


def build_labels(payload: dict, article: TrainingArticle) -> ArticleLabels:
    """Validate and normalise a raw label payload into a training brief."""
    topic = normalize_optional_text(str(payload.get("topic") or "")) or (article.title or "")
    if not topic:
        raise ValueError("labeller returned no usable topic")

    raw_keywords = payload.get("keywords") or []
    if isinstance(raw_keywords, str):
        raw_keywords = [k.strip() for k in raw_keywords.split(",")]

    return ArticleLabels(
        topic=topic,
        # Publisher-assigned, not inferred.
        category=CATEGORY_NAMES.get(article.category_slug or "", article.category_slug),
        industry=normalize_industry(payload.get("industry")),
        audience=normalize_optional_text(payload.get("audience")) or "Business leaders",
        keywords=normalize_keywords(raw_keywords),
        length=length_for(article.word_count),  # type: ignore[arg-type]
        # From the text, not the URL: /en/ URLs on this site can hold Thai.
        language=detect_language(article.clean_markdown or "")
        if article.clean_markdown
        else language_for(article.url),  # type: ignore[arg-type]
    )


async def label_one(client: AsyncOpenAI, model: str, article: TrainingArticle) -> ArticleLabels:
    body = (article.clean_markdown or "")[:MAX_ARTICLE_CHARS]
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": LABELER_SYSTEM},
            {"role": "user", "content": f"Title: {article.title or ''}\n\n{body}"},
        ],
        # Labels should be reproducible: same article in, same brief out.
        temperature=0.0,
        max_tokens=400,
        response_format={"type": "json_object"},
    )
    return build_labels(parse_label_json(completion.choices[0].message.content or ""), article)


async def run_label(
    store: CorpusStore,
    client: AsyncOpenAI,
    model: str,
    *,
    limit: int = 1000,
    concurrency: int = CONCURRENCY,
) -> tuple[dict[str, int], list[TrainingArticle]]:
    """Label only articles whose content changed since they were last labelled."""
    pending = (await store.due_for_label(label_version=LABEL_VERSION))[:limit]
    stats = {"pending": len(pending), "labelled": 0, "failed": 0}
    done: list[TrainingArticle] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def _worker(article: TrainingArticle) -> None:
        async with semaphore:
            try:
                labels = await label_one(client, model, article)
            except Exception as exc:  # a single bad reply must not sink the run
                stats["failed"] += 1
                await store.record_error(article.url, f"label: {exc}")
                logger.warning("label_failed", extra={"url": article.url, "error": str(exc)})
                return
            assert article.cleaned_hash is not None  # guaranteed by due_for_label
            await store.record_labels(
                article.url,
                labels=labels,
                labelled_hash=article.cleaned_hash,
                label_version=LABEL_VERSION,
                labeler_model=model,
            )
            stats["labelled"] += 1
            done.append(article.model_copy(update={"labels": labels}))

    async with asyncio.TaskGroup() as group:
        for article in pending:
            group.create_task(_worker(article))
    return stats, done


def labeler_client(base_url: str, api_key: str | None) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=base_url.rstrip("/"), api_key=api_key or "not-needed", timeout=120.0
    )


@app.command()
def run(
    limit: int = typer.Option(1000, help="Maximum articles to label this run."),
    model: str | None = typer.Option(None, help="Overrides LABELER_MODEL."),
    dry_run: bool = typer.Option(False, help="Label one article and print it; write nothing."),
) -> None:
    """Label new/changed articles (or all of them after a LABEL_VERSION bump)."""

    async def _run() -> None:
        async with pipeline_context() as (settings, store, _):
            labeler_model = model or settings.labeler_model
            if not settings.labeler_base_url or not labeler_model:
                typer.echo("set LABELER_BASE_URL, LABELER_API_KEY and LABELER_MODEL in .env first")
                raise typer.Exit(code=1)
            client = labeler_client(settings.labeler_base_url, settings.labeler_api_key)

            if dry_run:
                pending = await store.due_for_label(label_version=LABEL_VERSION)
                if not pending:
                    typer.echo("nothing to label")
                    return
                labels = await label_one(client, labeler_model, pending[0])
                typer.echo(pending[0].url)
                typer.echo(json.dumps(labels.model_dump(), indent=2))
                typer.echo("(dry run; nothing written)")
                return

            async with recorded_run(store, "label") as stats:
                result, _ = await run_label(store, client, labeler_model, limit=limit)
                stats.update(result, model=labeler_model)
            echo_stats("label", stats)

    run_async(_run)


@app.command()
def audit(sample: int = 10, seed: int = 7) -> None:
    """Print (article, inferred brief) pairs for human review.

    Label quality is judged by reading, not by a metric — a brief can be
    perfectly well-formed and still describe the wrong article.
    """

    async def _run() -> None:
        import random
        import textwrap

        async with pipeline_context() as (_, store, _):
            labelled = [a for a in await store.all() if a.labels]
        if not labelled:
            typer.echo("no labelled articles yet")
            return

        for article in random.Random(seed).sample(labelled, min(sample, len(labelled))):
            labels = article.labels
            assert labels is not None
            typer.echo("=" * 78)
            typer.echo(f"{article.url}  ({article.word_count} words)")
            typer.echo(f"  title    : {article.title}")
            typer.echo(f"  topic    : {labels.topic}")
            typer.echo(f"  category : {labels.category}")
            typer.echo(f"  industry : {labels.industry}")
            typer.echo(f"  audience : {labels.audience}")
            typer.echo(f"  keywords : {', '.join(labels.keywords)}")
            typer.echo(f"  length   : {labels.length}")
            excerpt = (article.clean_markdown or "")[:400].replace("\n", " ")
            typer.echo(
                textwrap.fill(f"  excerpt  : {excerpt}…", width=78, subsequent_indent="    ")
            )

    run_async(_run)


if __name__ == "__main__":
    app()
