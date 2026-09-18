"""Cleaning rules — each test pins one rule that protects the fine-tune."""

from __future__ import annotations

from pipeline.clean import (
    clean_markdown,
    count_words,
    drop_leading_title,
    drop_sections,
    find_duplicates,
    jaccard,
    normalize_headings,
    shingles,
    strip_boilerplate,
)
from pipeline.schemas import TrainingArticle

# Shaped like real trafilatura output from jenosize.com: h4 sections, h5
# sub-sections, bold inside headings, a trailing CTA and a "Loading..." artefact.
SAMPLE = """#### OKR vs KPI: Which One Fits Your Organization?

Measurement frameworks decide what a team optimises for.

#### **Understanding the Basics**

Both frameworks answer different questions.

##### **What is OKR?**

Objectives and Key Results pair ambition with measurement.

##### What is KPI?

Key Performance Indicators track steady-state health.

#### Choosing Between Them

Use OKR for change, KPI for stability.

Contact us today via our website to get started.

Loading...
"""


def test_headings_are_remapped_to_the_output_contract() -> None:
    # #### -> ##, ##### -> ###; the API's contract is "## for sections".
    result = normalize_headings(SAMPLE)
    assert "## OKR vs KPI: Which One Fits Your Organization?" in result
    assert "### What is OKR?" in result
    assert "####" not in result


def test_bold_markers_are_stripped_from_headings() -> None:
    assert normalize_headings("#### **Bold Heading**") == "## Bold Heading"


def test_body_emphasis_is_preserved() -> None:
    # Only headings are de-emphasised; prose keeps its formatting.
    assert "**important**" in normalize_headings("Some **important** prose.")


def test_boilerplate_and_cta_are_removed() -> None:
    result = strip_boilerplate(SAMPLE)
    assert "Loading" not in result
    assert "Contact us today" not in result
    assert "Objectives and Key Results" in result


def test_cta_removal_does_not_eat_headings() -> None:
    # A heading containing CTA-ish words is legitimate article structure.
    text = "## What to do next\n\nShip the pilot."
    assert "## What to do next" in strip_boilerplate(text)


def test_leading_title_is_dropped_when_it_restates_the_title() -> None:
    body = "## OKR vs KPI: Which One Fits Your Organization?\n\nBody text here."
    result = drop_leading_title(body, "OKR vs KPI: What's the Difference?")
    assert not result.startswith("##")
    assert result.startswith("Body text")


def test_leading_heading_is_kept_when_it_is_a_real_section() -> None:
    body = "## Why this matters now\n\nBody text."
    assert drop_leading_title(body, "OKR vs KPI: What's the Difference?").startswith("##")


def test_clean_markdown_end_to_end() -> None:
    result = clean_markdown(SAMPLE, "OKR vs KPI: What's the Difference?")
    assert "####" not in result
    assert "Loading" not in result
    assert "Contact us" not in result
    assert result.count("## ") >= 2
    assert not result.startswith("## OKR vs KPI: Which One")
    assert "\n\n\n" not in result  # blank runs collapsed


def test_count_words_ignores_markdown_syntax() -> None:
    assert count_words("## Heading\n\n**bold** text") == 3


def test_jaccard_bounds() -> None:
    assert jaccard(shingles("a b c d e f"), shingles("a b c d e f")) == 1.0
    assert jaccard(shingles("a b c d e f"), shingles("z y x w v u")) == 0.0
    assert jaccard(set(), shingles("a b c d e")) == 0.0


def _article(url: str, text: str) -> TrainingArticle:
    return TrainingArticle(url=url, clean_markdown=text, word_count=len(text.split()))


def test_find_duplicates_keeps_the_first_of_a_cluster() -> None:
    body = " ".join(f"word{i}" for i in range(80))
    articles = [
        _article("https://a", body),
        _article("https://b", body),  # exact copy
        _article("https://c", " ".join(f"other{i}" for i in range(80))),
    ]
    duplicates = find_duplicates(articles)
    assert duplicates == {"https://b"}


def test_find_duplicates_ignores_distinct_articles() -> None:
    articles = [
        _article("https://a", " ".join(f"alpha{i}" for i in range(80))),
        _article("https://b", " ".join(f"beta{i}" for i in range(80))),
    ]
    assert find_duplicates(articles) == set()


def test_cta_and_reference_sections_are_dropped_whole() -> None:
    # Taken from a real jenosize.com article that survived paragraph-level
    # CTA stripping with its heading and its bibliography intact.
    text = (
        "## Conclusion\n\nDelegate the tedious work.\n\n"
        "## Call to Action\n\nStart with one report.\n\n"
        "### References\n\n- Harvard Business Review: some study\n\n"
        "## After\n\nKept."
    )
    result = drop_sections(text)
    assert "## Conclusion" in result
    assert "Call to Action" not in result
    assert "Start with one report" not in result
    assert "Harvard" not in result  # the model must never learn to cite
    assert "## After" in result and "Kept." in result


def test_drop_sections_only_matches_whole_heading_prefixes() -> None:
    # "Sources of growth" is a real section, not a bibliography.
    assert "## Sourcing talent" in drop_sections("## Sourcing talent\n\nBody.")


def test_heading_remap_is_relative_to_the_shallowest_level() -> None:
    # Older jenosize.com articles start at <h5>; a fixed h5 -> ### map left them
    # with no ## sections at all, which dataset validation caught.
    result = normalize_headings("##### What are Megatrends?\n\n###### Detail\n\nBody.")
    assert "## What are Megatrends?" in result
    assert "### Detail" in result


def test_sections_stay_top_level_when_the_title_is_the_only_shallow_heading() -> None:
    # Real shape of older articles: #### restated title, then ##### sections.
    raw = "#### 7 Megatrends in 2024\n\nIntro.\n\n##### What are Megatrends?\n\nBody."
    result = clean_markdown(raw, "7 Megatrends in 2024 Worth Noting")
    assert result.startswith("Intro.")
    assert "## What are Megatrends?" in result
    assert "###" not in result
