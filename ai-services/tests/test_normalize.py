"""Table-driven coverage of the normalization rules."""

from __future__ import annotations

import pytest

from app.core.errors import ValidationError
from app.schemas.articles import ArticleRequest
from app.services.normalize import (
    DEFAULT_AUDIENCE,
    MAX_KEYWORDS,
    collapse_whitespace,
    normalize_industry,
    normalize_keywords,
    normalize_request,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  hello   world  ", "hello world"),
        ("line\n\nbreak", "line break"),
        ("tab\tsep", "tab sep"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_collapse_whitespace(raw: str, expected: str) -> None:
    assert collapse_whitespace(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(["AI", "ai", " Ai "], ["ai"], id="dedupe-case-and-space"),
        pytest.param(["B", "A", "C"], ["b", "a", "c"], id="order-preserved"),
        pytest.param(["", "  ", "x"], ["x"], id="drop-empties"),
        pytest.param(
            [f"k{i}" for i in range(15)], [f"k{i}" for i in range(MAX_KEYWORDS)], id="cap-at-10"
        ),
        pytest.param(None, [], id="none"),
        pytest.param(["multi   word  key"], ["multi word key"], id="inner-whitespace"),
    ],
)
def test_normalize_keywords(raw: list[str] | None, expected: list[str]) -> None:
    assert normalize_keywords(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("fintech", "Financial Services", id="synonym"),
        pytest.param("FinTech", "Financial Services", id="synonym-cased"),
        pytest.param("ecommerce", "Retail & E-commerce", id="ecommerce"),
        pytest.param("E-Commerce", "Retail & E-commerce", id="punctuated"),
        pytest.param("Retail & E-commerce", "Retail & E-commerce", id="already-canonical"),
        pytest.param("supply chain", "Logistics & Supply Chain", id="multiword"),
        pytest.param("space tourism", "Space Tourism", id="unknown-title-cased"),
        pytest.param("   ", None, id="blank-to-none"),
        pytest.param(None, None, id="none"),
    ],
)
def test_normalize_industry(raw: str | None, expected: str | None) -> None:
    assert normalize_industry(raw) == expected


def test_normalize_request_applies_every_rule() -> None:
    request = ArticleRequest(
        topic="  The  Future of   Payments ",
        category=" Futurist ",
        industry="fintech",
        audience="   ",
        keywords="Embedded Finance, embedded finance ,  BNPL ",  # type: ignore[arg-type]  # coerced by the validator
        length="long",
    )
    params = normalize_request(request)

    assert params.topic == "The Future of Payments"
    assert params.category == "Futurist"
    assert params.industry == "Financial Services"
    assert params.audience == DEFAULT_AUDIENCE  # blank audience falls back
    assert params.keywords == ["embedded finance", "bnpl"]
    assert params.target_words == 1500  # long


def test_normalize_request_rejects_whitespace_only_topic() -> None:
    # min_length=3 lets "   " past pydantic, so the service must reject it.
    with pytest.raises(ValidationError):
        normalize_request(ArticleRequest(topic="    "))
