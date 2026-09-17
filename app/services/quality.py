"""Deterministic quality gate on generated articles.

An LLM judge would be more nuanced but costs a second GPU round-trip and is not
reproducible. These four checks catch the failure modes that actually show up
with a small fine-tuned model: a stub-length article, no structure, a headline
that will be truncated in search results, and ignored SEO keywords.

`generation.py` uses `QualityReport.warnings` as feedback for exactly one retry.
"""

from __future__ import annotations

import re

from app.schemas.articles import NormalizedParams, QualityReport

# SEO convention: Google truncates titles past roughly 60 characters.
TITLE_MIN, TITLE_MAX = 25, 70
MIN_HEADINGS = 3
# A short article is acceptable; a *much* shorter one than requested is not.
WORD_COUNT_TOLERANCE = 0.35
MIN_KEYWORD_COVERAGE = 0.7

_H2 = re.compile(r"^##\s+\S", re.MULTILINE)
_MD_NOISE = re.compile(r"[#*_`>\[\]()]|^\s*[-+]\s+", re.MULTILINE)


def count_words(markdown: str) -> int:
    """Word count of the prose, with markdown syntax stripped so it isn't inflated."""
    return len(_MD_NOISE.sub(" ", markdown).split())


def count_h2(markdown: str) -> int:
    return len(_H2.findall(markdown))


def keyword_coverage(markdown: str, keywords: list[str]) -> float:
    """Fraction of requested keywords that appear in the body (case-insensitive).

    Substring matching is intentional: "agentic ai" should count as covered by
    "agentic AI-driven", which a token-equality check would miss.
    """
    if not keywords:
        return 1.0
    haystack = markdown.lower()
    hits = sum(1 for kw in keywords if kw.lower() in haystack)
    return hits / len(keywords)


def evaluate(markdown: str, title: str, params: NormalizedParams) -> QualityReport:
    words = count_words(markdown)
    headings = count_h2(markdown)
    coverage = keyword_coverage(markdown, params.keywords)
    title_len = len(title.strip())

    warnings: list[str] = []
    floor = int(params.target_words * (1 - WORD_COUNT_TOLERANCE))
    if words < floor:
        warnings.append(
            f"Article is {words} words; expand to at least {floor} (target {params.target_words})."
        )
    if headings < MIN_HEADINGS:
        warnings.append(f"Only {headings} H2 sections; use at least {MIN_HEADINGS}.")
    if coverage < MIN_KEYWORD_COVERAGE:
        missing = [kw for kw in params.keywords if kw.lower() not in markdown.lower()]
        warnings.append("Missing SEO keywords: " + ", ".join(missing) + ".")
    if not TITLE_MIN <= title_len <= TITLE_MAX:
        warnings.append(
            f"Title is {title_len} characters; keep it between {TITLE_MIN} and {TITLE_MAX}."
        )

    return QualityReport(
        keyword_coverage=round(coverage, 4),
        word_count=words,
        heading_count=headings,
        title_length=title_len,
        passed=not warnings,
        warnings=warnings,
    )
