"""Dataset construction.

The first test here is the one that matters most in the whole repo: if the
training prompt and the inference prompt ever diverge, the fine-tune is
optimised for a prompt the service never sends, and no downstream metric would
tell you.
"""

from __future__ import annotations

import json

import pytest

from app.schemas.articles import ArticleRequest
from app.services import normalize
from app.services import prompt as prompt_service
from app.services.llm import parse_article
from pipeline.build_dataset import eligible, meta_for, params_for, to_example
from pipeline.schemas import ArticleLabels, TrainingArticle

BODY = (
    "Payments infrastructure is being rebuilt around programmable money.\n\n"
    "## Why this matters now\n\nThe shift is visible in operating metrics.\n\n"
    "## Where value shows up\n\nSettlement latency is the first line item.\n\n"
    "## How to act\n\nInstrument one corridor end to end."
)


def _article(**overrides: object) -> TrainingArticle:
    base: dict[str, object] = {
        "url": "https://www.jenosize.com/en/ideas/futurist/embedded-finance",
        "category_slug": "futurist",
        "title": "Embedded Finance: What Comes Next for Banks",
        "meta_description": "How embedded finance reshapes bank distribution.",
        "clean_markdown": BODY,
        "word_count": 600,
        "labels": ArticleLabels(
            topic="The future of embedded finance",
            category="Futurist",
            industry="Financial Services",
            audience="Banking executives",
            keywords=["embedded finance", "programmable money"],
            length="short",
        ),
    }
    return TrainingArticle(**{**base, **overrides})  # type: ignore[arg-type]


def test_training_prompt_is_byte_identical_to_the_inference_prompt() -> None:
    """The invariant the whole fine-tune rests on.

    Build the prompt the way the *pipeline* does, and the way the *API* does for
    an equivalent user request, and require that they match exactly.
    """
    article = _article()
    example = to_example(article)

    labels = article.labels
    assert labels is not None
    api_params = normalize.normalize_request(
        ArticleRequest(
            topic=labels.topic,
            category=labels.category,
            industry=labels.industry,
            audience=labels.audience,
            keywords=labels.keywords,
            length=labels.length,
            language=labels.language,
        )
    )

    assert example["messages"][0]["content"] == prompt_service.build_system_prompt(api_params)
    assert example["messages"][1]["content"] == prompt_service.build_user_prompt(api_params, [])


def test_labels_survive_a_round_trip_through_the_api_normalizer() -> None:
    # Labels were already normalised at labelling time, so re-normalising them
    # must be a no-op — otherwise training and inference drift by one pass.
    article = _article()
    assert article.labels is not None
    pipeline_params = params_for(article)
    api_params = normalize.normalize_request(
        ArticleRequest(
            **article.labels.model_dump(exclude={"category"}), category=article.labels.category
        )
    )
    assert pipeline_params == api_params


def test_assistant_turn_is_parseable_by_the_production_parser() -> None:
    example = to_example(_article())
    title, meta, body = parse_article(example["messages"][2]["content"])
    assert title == "Embedded Finance: What Comes Next for Banks"
    assert meta == "How embedded finance reshapes bank distribution."
    assert body.startswith("Payments infrastructure")
    assert "## Why this matters now" in body


def test_training_examples_contain_no_fabricated_sources() -> None:
    # Corpus articles were not written against supplied sources, so inventing a
    # "Reference material" block would teach the model to ignore the real one.
    user_turn = to_example(_article())["messages"][1]["content"]
    assert "Reference material" not in user_turn
    assert "No reference material was supplied" in user_turn


def test_meta_falls_back_to_the_opening_paragraph() -> None:
    meta = meta_for(_article(meta_description=None))
    assert meta.startswith("Payments infrastructure")
    assert len(meta) <= 155


def test_meta_fallback_never_returns_empty() -> None:
    assert meta_for(_article(meta_description=None, clean_markdown="Short.")) == "Short."


def test_meta_prefers_the_publishers_own_description() -> None:
    assert meta_for(_article()) == "How embedded finance reshapes bank distribution."


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        pytest.param({"labels": None}, "unlabelled", id="no-labels"),
        pytest.param({"clean_markdown": None}, "uncleaned", id="no-markdown"),
        pytest.param({"title": None}, "no title to train TITLE: on", id="no-title"),
        pytest.param({"is_duplicate": True}, "near-duplicate", id="duplicate"),
    ],
)
def test_ineligible_articles_are_excluded(overrides: dict, reason: str) -> None:
    assert eligible([_article(**overrides)]) == [], reason


def test_eligible_keeps_a_complete_article() -> None:
    assert len(eligible([_article()])) == 1


def test_example_carries_traceable_metadata() -> None:
    example = to_example(_article())
    assert example["meta"]["url"].endswith("embedded-finance")
    assert example["meta"]["word_count"] == 600
    # The whole row must be JSON-serialisable: it is written as one JSONL line.
    assert json.loads(json.dumps(example)) == example
