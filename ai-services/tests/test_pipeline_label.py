"""Reverse-labelling: parsing tolerance and normalization guarantees."""

from __future__ import annotations

import pytest

from pipeline.label import (
    CATEGORY_NAMES,
    build_labels,
    language_for,
    length_for,
    parse_label_json,
)
from pipeline.schemas import TrainingArticle


def _article(**overrides: object) -> TrainingArticle:
    base = {
        "url": "https://www.jenosize.com/en/ideas/futurist/agentic-ai",
        "category_slug": "futurist",
        "title": "Agentic AI",
        "clean_markdown": "## Body",
        "word_count": 1000,
    }
    return TrainingArticle(**{**base, **overrides})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param('{"topic": "x"}', id="bare"),
        pytest.param('```json\n{"topic": "x"}\n```', id="fenced"),
        pytest.param(
            'Sure! Here is the brief:\n{"topic": "x"}\nHope that helps.', id="prose-wrapped"
        ),
        pytest.param('  \n{"topic": "x"}\n  ', id="whitespace"),
    ],
)
def test_parse_label_json_tolerates_model_chatter(raw: str) -> None:
    # Small models wrap JSON even when told not to; a whole run must not die on it.
    assert parse_label_json(raw) == {"topic": "x"}


def test_parse_label_json_rejects_non_json() -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        parse_label_json("I could not do that.")


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        (400, "short"),
        (799, "short"),
        (800, "medium"),
        (1249, "medium"),
        (1250, "long"),
        (3000, "long"),
    ],
)
def test_length_is_computed_not_inferred(words: int, expected: str) -> None:
    assert length_for(words) == expected


def test_language_comes_from_the_url() -> None:
    assert language_for("https://www.jenosize.com/en/ideas/futurist/x") == "en"
    assert language_for("https://www.jenosize.com/th/ideas/futurist/x") == "th"


def test_build_labels_normalises_through_the_api_vocabulary() -> None:
    # The whole point: training labels must use the same strings the API
    # produces from a live request.
    labels = build_labels(
        {
            "topic": "  The  Future of  Payments ",
            "industry": "fintech",
            "audience": " CFOs ",
            "keywords": ["Embedded Finance", "embedded finance", "BNPL"],
        },
        _article(word_count=600),
    )
    assert labels.topic == "The Future of Payments"
    assert labels.industry == "Financial Services"  # synonym -> canonical
    assert labels.audience == "CFOs"
    assert labels.keywords == ["embedded finance", "bnpl"]  # deduped, lowercased
    assert labels.length == "short"  # from word_count, not the model


def test_category_is_taken_from_the_url_not_the_model() -> None:
    labels = build_labels(
        {"topic": "x", "category": "Something The Model Invented"},
        _article(category_slug="real-time-marketing"),
    )
    assert labels.category == CATEGORY_NAMES["real-time-marketing"]


def test_unknown_category_slug_passes_through() -> None:
    labels = build_labels({"topic": "x"}, _article(category_slug="brand-new-section"))
    assert labels.category == "brand-new-section"


def test_missing_topic_falls_back_to_the_title() -> None:
    assert build_labels({}, _article(title="A Real Headline")).topic == "A Real Headline"


def test_no_topic_and_no_title_is_an_error() -> None:
    with pytest.raises(ValueError, match="no usable topic"):
        build_labels({}, _article(title=None))


def test_missing_audience_gets_the_house_default() -> None:
    assert build_labels({"topic": "x"}, _article()).audience == "Business leaders"


def test_keywords_accept_a_comma_separated_string() -> None:
    labels = build_labels({"topic": "x", "keywords": "a, b, a"}, _article())
    assert labels.keywords == ["a", "b"]
