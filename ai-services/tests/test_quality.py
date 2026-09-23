"""Quality-gate thresholds."""

from __future__ import annotations

from app.schemas.articles import NormalizedParams
from app.services.quality import count_h2, count_words, evaluate, keyword_coverage


def _params(**overrides: object) -> NormalizedParams:
    base = {
        "topic": "Retail trends",
        "keywords": ["retail media", "personalization"],
        "length": "short",
        "target_words": 100,
    }
    return NormalizedParams(**{**base, **overrides})  # type: ignore[arg-type]


def _article(words: int = 200, headings: int = 3, keywords: tuple[str, ...] = ()) -> str:
    body = " ".join(["word"] * words)
    sections = "\n".join(f"## Section {i}\n\n{body}\n" for i in range(headings))
    return sections + "\n" + " ".join(keywords)


def test_count_words_ignores_markdown_syntax() -> None:
    assert count_words("## Heading\n\n**bold** text") == 3


def test_count_h2_only_counts_h2() -> None:
    assert count_h2("# Title\n## A\n### B\n## C\n") == 2


def test_keyword_coverage_is_substring_based() -> None:
    assert keyword_coverage("agentic AI-driven retail", ["agentic ai"]) == 1.0
    assert keyword_coverage("nothing here", ["agentic ai"]) == 0.0
    assert keyword_coverage("only one of two", ["only", "missing"]) == 0.5


def test_keyword_coverage_is_one_when_no_keywords_requested() -> None:
    assert keyword_coverage("whatever", []) == 1.0


def test_good_article_passes() -> None:
    markdown = _article(words=200, headings=3, keywords=("retail media", "personalization"))
    report = evaluate(markdown, "A Perfectly Reasonable Headline Here", _params())
    assert report.passed
    assert report.warnings == []
    assert report.keyword_coverage == 1.0


def test_short_article_is_flagged() -> None:
    markdown = _article(words=5, headings=3, keywords=("retail media", "personalization"))
    report = evaluate(markdown, "A Perfectly Reasonable Headline Here", _params())
    assert not report.passed
    assert any("expand" in w for w in report.warnings)


def test_missing_headings_flagged() -> None:
    markdown = _article(words=200, headings=1, keywords=("retail media", "personalization"))
    report = evaluate(markdown, "A Perfectly Reasonable Headline Here", _params())
    assert any("H2" in w for w in report.warnings)


def test_missing_keywords_are_named() -> None:
    markdown = _article(words=200, headings=3, keywords=("retail media",))
    report = evaluate(markdown, "A Perfectly Reasonable Headline Here", _params())
    assert any("personalization" in w for w in report.warnings)


def test_title_length_bounds() -> None:
    markdown = _article(words=200, headings=3, keywords=("retail media", "personalization"))
    assert any("characters" in w for w in evaluate(markdown, "Too short", _params()).warnings)
    assert any("characters" in w for w in evaluate(markdown, "x" * 120, _params()).warnings)
