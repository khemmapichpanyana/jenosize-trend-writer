"""Step 2 — turn archived HTML into training-grade markdown.

    uv run python -m pipeline.clean run
    uv run python -m pipeline.clean stats

The cleaning rules decide what the model learns to imitate, so they are the
highest-leverage code in the data pipeline. Every rule below exists to stop the
fine-tune from learning something we do not want it to reproduce.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import typer

from app.core.logging import get_logger
from app.storage.base import Storage
from pipeline._cli import echo_stats, pipeline_context, recorded_run, run_async
from pipeline.schemas import TrainingArticle
from pipeline.store import CorpusStore

logger = get_logger(__name__)
app = typer.Typer(help="Clean archived HTML into training markdown.")

# The API's output contract is "## for sections, ### for sub-sections", and a
# fine-tune trained on anything else would emit it forever. The site is not
# consistent: newer articles use <h4>/<h5>, older ones start at <h5>. So the
# remap is *relative* — each article's shallowest heading level becomes ##.
SECTION_LEVEL, SUBSECTION_LEVEL = 2, 3

# Lines that are site furniture rather than article prose.
BOILERPLATE_LINES = re.compile(
    r"^\s*(loading\.{0,3}|share this article|related articles?|previous|next|"
    r"read more|tags?:.*|\d+\s*min read|subscribe.*|follow us.*)\s*$",
    re.IGNORECASE,
)

# The house CTA. Kept out of training data because a model that learned it would
# append "contact Jenosize" to an article written *for* Jenosize's own client.
# The output contract still asks for a closing "next step" — just not an advert.
CTA_PATTERNS = re.compile(
    r"(contact us (today|now)|get in touch|via our website|jenosize (can help|is ready)|"
    r"let'?s (talk|work together)|book a (demo|consultation))",
    re.IGNORECASE,
)

# Whole sections that must not be learned, dropped heading-and-body:
#   * CTA sections — same reason as CTA_PATTERNS, at section granularity;
#   * reference lists — a model trained on them learns to *write* citations,
#     which at inference means fabricating sources it was never given. Facts
#     come from retrieval; the adapter must never invent a bibliography.
DROP_SECTIONS = re.compile(
    r"^(call to action|cta|references?|sources?|bibliography|further reading|"
    r"related (articles?|reading)|read more|about (the )?author|about jenosize)\b",
    re.IGNORECASE,
)

# Bump whenever a rule in this file changes. Every article whose clean_version
# differs is re-cleaned on the next run (from the archived raw HTML in R2, so no
# re-crawl); nothing else is touched.
CLEAN_VERSION = 1

MIN_WORDS = 300
MAX_WORDS = 4000
# Two articles sharing this fraction of their 5-word shingles are near-copies.
DUPLICATE_THRESHOLD = 0.6
SHINGLE_SIZE = 5

_HEADING = re.compile(r"^(#{1,6})\s*(.+?)\s*#*$")
_EMPHASIS = re.compile(r"(\*\*|__|\*|_)")
_BLANKS = re.compile(r"\n{3,}")
_WORD = re.compile(r"[A-Za-z0-9'฀-๿]+")


def normalize_headings(markdown: str) -> str:
    """Remap heading levels onto the `##`/`###` contract and de-emphasise them.

    The shallowest level present becomes `##`; everything deeper becomes `###`
    (two levels is all the output contract uses). Headings arrive as
    `#### **What is OKR?**`; the bold markers are visual noise that would teach
    the model to emit them inside headings.
    """
    levels = [len(m.group(1)) for line in markdown.splitlines() if (m := _HEADING.match(line))]
    top = min(levels, default=SECTION_LEVEL)

    out: list[str] = []
    for line in markdown.splitlines():
        match = _HEADING.match(line)
        if not match:
            out.append(line)
            continue
        level = SECTION_LEVEL if len(match.group(1)) == top else SUBSECTION_LEVEL
        text = _EMPHASIS.sub("", match.group(2)).strip()
        out.append(f"{'#' * level} {text}" if text else "")
    return "\n".join(out)


def strip_boilerplate(markdown: str) -> str:
    """Drop site furniture and the house call-to-action."""
    kept: list[str] = []
    for block in re.split(r"\n\s*\n", markdown):
        stripped = block.strip()
        if not stripped:
            continue
        if BOILERPLATE_LINES.match(stripped):
            continue
        # Only paragraphs are dropped for CTA language; a heading that happens
        # to contain "next" is legitimate article structure.
        if not stripped.startswith("#") and CTA_PATTERNS.search(stripped):
            continue
        kept.append(stripped)
    return "\n\n".join(kept)


def drop_sections(markdown: str) -> str:
    """Remove DROP_SECTIONS headings and everything under them.

    A dropped section ends at the next heading of the same or a higher level,
    so a `## References` block takes its `###` children with it but leaves the
    following `##` section alone.
    """
    out: list[str] = []
    dropping_level: int | None = None
    for line in markdown.splitlines():
        heading = _HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            if dropping_level is not None and level <= dropping_level:
                dropping_level = None
            if dropping_level is None and DROP_SECTIONS.match(heading.group(2).strip()):
                dropping_level = level
                continue
        if dropping_level is None:
            out.append(line)
    return "\n".join(out)


def drop_leading_title(markdown: str, title: str | None) -> str:
    """Remove the opening heading when it just restates the article title.

    The title is a separate field in the training target (`TITLE:`), so leaving
    a duplicate at the top of the body teaches the model to emit it twice.
    """
    lines = markdown.lstrip().splitlines()
    if not lines or not lines[0].startswith("#"):
        return markdown
    heading = _HEADING.match(lines[0])
    if not heading:
        return markdown
    first = heading.group(2)
    if title and _similar(first, title) < 0.4:
        return markdown
    return "\n".join(lines[1:]).lstrip()


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text)]


def _similar(a: str, b: str) -> float:
    """Token-overlap ratio; good enough to spot a restated headline."""
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def count_words(markdown: str) -> int:
    return len(_WORD.findall(re.sub(r"[#*`>\[\]()]", " ", markdown)))


def clean_markdown(raw_markdown: str, title: str | None = None) -> str:
    """Apply every rule, in the order they depend on each other.

    The restated title goes first: it is often the article's only shallow
    heading, and if it were still present when levels are computed, every real
    section would be demoted to `###`.
    """
    text = drop_leading_title(raw_markdown, title)
    text = normalize_headings(text)
    text = drop_sections(text)
    text = strip_boilerplate(text)
    return _BLANKS.sub("\n\n", text).strip()


def shingles(text: str, size: int = SHINGLE_SIZE) -> set[str]:
    words = _tokens(text)
    return {" ".join(words[i : i + size]) for i in range(max(0, len(words) - size + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_duplicates(articles: Iterable[TrainingArticle]) -> set[str]:
    """URLs of near-duplicate articles, keeping the first of each cluster.

    O(n^2) over the corpus, which is fine at a few hundred articles and far
    easier to reason about than MinHash. Revisit if the corpus reaches 10k.
    """
    seen: list[tuple[str, set[str]]] = []
    duplicates: set[str] = set()
    for article in articles:
        if not article.clean_markdown:
            continue
        fingerprint = shingles(article.clean_markdown)
        if any(jaccard(fingerprint, other) >= DUPLICATE_THRESHOLD for _, other in seen):
            duplicates.add(article.url)
            continue
        seen.append((article.url, fingerprint))
    return duplicates


async def run_clean(
    store: CorpusStore,
    storage: Storage,
    *,
    min_words: int = MIN_WORDS,
    max_words: int = MAX_WORDS,
) -> dict[str, int]:
    """Clean only articles whose content or cleaning rules changed since last time."""
    import trafilatura

    pending = await store.due_for_clean(clean_version=CLEAN_VERSION)
    stats = {"pending": len(pending), "cleaned": 0, "too_short": 0, "too_long": 0, "failed": 0}

    for article in pending:
        assert article.content_hash is not None  # guaranteed by due_for_clean
        markdown: str | None = None
        words = 0
        error: str | None = None
        try:
            raw = await storage.get_bytes(article.r2_raw_key or "")
            extracted = trafilatura.extract(
                raw.decode("utf-8", errors="replace"),
                url=article.url,
                output_format="markdown",
                include_formatting=True,
                include_tables=True,
                include_comments=False,
                favor_precision=True,
            )
            if not extracted:
                raise ValueError("no text extracted")
            body = clean_markdown(extracted, article.title)
            words = count_words(body)
            if words < min_words:
                stats["too_short"] += 1
                error = f"clean: too short ({words} words)"
            elif words > max_words:
                stats["too_long"] += 1
                error = f"clean: too long ({words} words)"
            else:
                stats["cleaned"] += 1
                markdown = body
        except Exception as exc:  # one bad page must not stop the run
            stats["failed"] += 1
            error = f"clean: {exc}"
            logger.warning("clean_failed", extra={"url": article.url, "error": str(exc)})

        await store.record_clean(
            article.url,
            content_hash=article.content_hash,
            clean_version=CLEAN_VERSION,
            clean_markdown=markdown,
            word_count=words,
            error=error,
        )

    # Duplicates are a corpus-wide property: a new article can duplicate an old
    # one, so this pass always covers everything, not just this batch. It only
    # reads markdown already in Postgres, so it is cheap.
    duplicates = find_duplicates(await store.cleaned())
    await store.set_duplicates(duplicates)
    stats["near_duplicates"] = len(duplicates)
    return stats


@app.command()
def run(min_words: int = MIN_WORDS, max_words: int = MAX_WORDS) -> None:
    """Clean new/changed articles (or all of them after a CLEAN_VERSION bump)."""

    async def _run() -> None:
        async with pipeline_context() as (_, store, storage):
            async with recorded_run(store, "clean") as stats:
                stats.update(
                    await run_clean(store, storage, min_words=min_words, max_words=max_words)
                )
            echo_stats("clean", stats)

    run_async(_run)


@app.command()
def stats(json_out: bool = typer.Option(False, "--json", help="Machine-readable output.")) -> None:
    """Corpus health — the numbers that populate the data card."""

    async def _run() -> None:
        import json
        import statistics

        async with pipeline_context() as (_, store, _):
            articles = await store.all()
        usable = [a for a in articles if a.clean_markdown and not a.is_duplicate]
        counts = [a.word_count for a in usable]

        by_category: dict[str, int] = {}
        for article in usable:
            key = article.category_slug or "unknown"
            by_category[key] = by_category.get(key, 0) + 1
        by_category = dict(sorted(by_category.items(), key=lambda kv: -kv[1]))

        payload: dict[str, object] = {
            "discovered": len(articles),
            "usable": len(usable),
            "duplicates": sum(1 for a in articles if a.is_duplicate),
            "errors": sum(1 for a in articles if a.error),
            "words_min": min(counts) if counts else 0,
            "words_median": int(statistics.median(counts)) if counts else 0,
            "words_mean": int(statistics.mean(counts)) if counts else 0,
            "words_max": max(counts) if counts else 0,
            "words_total": sum(counts),
            "by_category": by_category,
        }
        if json_out:
            typer.echo(json.dumps(payload, indent=2))
            return
        typer.echo(f"usable articles : {payload['usable']} / {payload['discovered']} discovered")
        typer.echo(f"near-duplicates : {payload['duplicates']}")
        typer.echo(f"errors          : {payload['errors']}")
        typer.echo(
            f"words           : min {payload['words_min']} · median {payload['words_median']}"
            f" · mean {payload['words_mean']} · max {payload['words_max']}"
            f" · total {payload['words_total']:,}"
        )
        typer.echo("by category:")
        for category, count in by_category.items():
            typer.echo(f"  {count:>4}  {category}")

    run_async(_run)


if __name__ == "__main__":
    app()
