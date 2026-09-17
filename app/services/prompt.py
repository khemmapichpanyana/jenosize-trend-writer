"""Prompt construction.

The system prompt is the *style contract*. Day 0 ships placeholder house rules
derived from a skim of Jenosize Ideas; Day 1's labeling pass replaces them with
rules extracted from the real corpus, and Day 2 bakes most of them into the LoRA
adapter so the system prompt can shrink (a shorter prompt is cheaper and drifts
less).

Output format is a fenced contract rather than JSON mode because vLLM streams
plain text token-by-token, and the demo needs to render markdown as it arrives.
"""

from __future__ import annotations

from app.schemas.articles import NormalizedParams
from app.services.retrieve import Chunk

# TODO(day-1): regenerate these rules from the labeled Jenosize corpus.
STYLE_RULES = """You write for Jenosize Ideas, a Thai business-transformation consultancy.

House style:
- Executive, forward-looking and concrete. No hype, no filler, no "in today's fast-paced world".
- Lead with the business implication, then the evidence, then the action.
- Second person plural ("your organisation") when addressing the reader; never "I".
- Prefer specifics over adjectives: name technologies, markets, timeframes.
- Southeast Asian context where relevant, without forcing it.
- Do not invent statistics, company names, dates or quotes. If the provided
  sources do not support a number, describe the direction of change instead.
"""

OUTPUT_CONTRACT = """Return exactly this structure and nothing else:

TITLE: <one headline, 25-70 characters>
META: <meta description, 120-160 characters>
---
<article body in Markdown, starting with a one-paragraph hook, then ## sections>

Body rules:
- Use `##` for section headings. Do not use `#` (the title is separate).
- At least 3 `##` sections plus a closing section with a concrete next step.
- Weave every SEO keyword in naturally at least once.
"""


def build_system_prompt(params: NormalizedParams) -> str:
    language = "Thai" if params.language == "th" else "English"
    return f"{STYLE_RULES}\nWrite in {language}.\n\n{OUTPUT_CONTRACT}"


def format_sources(chunks: list[Chunk]) -> str:
    """Render retrieved chunks as numbered evidence the model may draw on."""
    if not chunks:
        return ""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        label = chunk.source_title or chunk.source_url or f"Source {i}"
        blocks.append(f"[{i}] {label}\n{chunk.text.strip()}")
    return (
        "Reference material (use only what is relevant; do not cite anything not listed):\n\n"
        + "\n\n".join(blocks)
    )


def build_user_prompt(
    params: NormalizedParams,
    chunks: list[Chunk] | None = None,
    *,
    feedback: list[str] | None = None,
) -> str:
    """Assemble the brief. `feedback` is the quality report from a failed attempt."""
    lines = [f"Topic: {params.topic}"]
    if params.category:
        lines.append(f"Category: {params.category}")
    if params.industry:
        lines.append(f"Industry: {params.industry}")
    lines.append(f"Target audience: {params.audience}")
    if params.keywords:
        lines.append(f"SEO keywords: {', '.join(params.keywords)}")
    lines.append(f"Target length: about {params.target_words} words")

    prompt = "Write a Jenosize Ideas article.\n\n" + "\n".join(lines)

    if sources := format_sources(chunks or []):
        prompt += f"\n\n{sources}"
    else:
        # Said explicitly so the model does not hallucinate citations it never got.
        prompt += "\n\nNo reference material was supplied. Do not cite sources or invent figures."

    if feedback:
        prompt += "\n\nYour previous draft was rejected. Fix all of these and rewrite in full:\n"
        prompt += "\n".join(f"- {item}" for item in feedback)

    return prompt
