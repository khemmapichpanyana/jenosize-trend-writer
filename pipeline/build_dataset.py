"""Step 4 — emit the chat JSONL that `modal/train.py` consumes.

    uv run python -m pipeline.build_dataset run --version v1
    uv run python -m pipeline.build_dataset validate --version v1

CRITICAL INVARIANT — prompt parity
----------------------------------
The training prompt is produced by `app.services.prompt.build_system_prompt` and
`build_user_prompt`: the *same functions the API calls at inference time*. If
this file grew its own template, the adapter would be optimised for a prompt the
service never sends, and the fine-tune would silently underperform in a way no
metric here would catch. `tests/test_pipeline_dataset.py` asserts the parity.

Output, one JSON object per line:

    {"messages": [
        {"role": "system",    "content": "<style rules + output contract>"},
        {"role": "user",      "content": "<rendered brief>"},
        {"role": "assistant", "content": "TITLE: …\\nMETA: …\\n---\\n<markdown>"}
    ]}

The assistant turn reproduces the output contract exactly, so the model learns
to emit the format `app.services.llm.parse_article` already knows how to read.
"""

from __future__ import annotations

import json
from typing import Any

import typer

from app.schemas.articles import TARGET_WORDS, NormalizedParams
from app.services import prompt as prompt_service
from app.services.llm import parse_article
from app.storage import put_text
from app.storage.keys import dataset_key
from pipeline._cli import context, run_async
from pipeline.schemas import Split, TrainingArticle
from pipeline.store import split_for

app = typer.Typer(help="Build train/eval JSONL from labelled articles.")

# A meta description is part of the output contract, so a missing one has to be
# synthesised rather than left blank — otherwise the model learns to skip it.
META_FALLBACK_CHARS = 155


def params_for(article: TrainingArticle) -> NormalizedParams:
    """The labels, as the exact object the API builds from a live request."""
    labels = article.labels
    assert labels is not None, "params_for requires a labelled article"
    return NormalizedParams(
        topic=labels.topic,
        category=labels.category,
        industry=labels.industry,
        audience=labels.audience or "Business leaders",
        keywords=labels.keywords,
        length=labels.length,
        language=labels.language,
        target_words=TARGET_WORDS[labels.length],
    )


def meta_for(article: TrainingArticle) -> str:
    if article.meta_description:
        return article.meta_description.strip()
    first = (article.clean_markdown or "").strip().split("\n\n")[0]
    text = first.replace("\n", " ").strip()
    return text[:META_FALLBACK_CHARS].rsplit(" ", 1)[0] if len(text) > META_FALLBACK_CHARS else text


def to_example(article: TrainingArticle) -> dict[str, Any]:
    """Render one article into a chat-format training example."""
    params = params_for(article)
    # No retrieved chunks: the corpus articles were not written against sources,
    # so training on a fabricated "Reference material" block would teach the
    # model to ignore the one we *do* send at inference time.
    return {
        "messages": [
            {"role": "system", "content": prompt_service.build_system_prompt(params)},
            {"role": "user", "content": prompt_service.build_user_prompt(params, [])},
            {
                "role": "assistant",
                "content": (
                    f"TITLE: {article.title}\n"
                    f"META: {meta_for(article)}\n"
                    f"---\n"
                    f"{(article.clean_markdown or '').strip()}\n"
                ),
            },
        ],
        # Carried for traceability; the trainer reads `messages` only.
        "meta": {"url": article.url, "word_count": article.word_count},
    }


def eligible(articles: list[TrainingArticle]) -> list[TrainingArticle]:
    return [a for a in articles if a.labels and a.clean_markdown and a.title and not a.is_duplicate]


@app.command()
def run(
    version: str = "v1",
    eval_frac: float = typer.Option(0.1, help="Fraction held out for evaluation."),
    seed: int = 13,
    out_dir: str = typer.Option(".data/datasets", help="Local output directory."),
    upload: bool = typer.Option(True, help="Also write to object storage."),
) -> None:
    """Write `datasets/{version}/train.jsonl`, `eval.jsonl` and `data_card.md`."""

    async def _run() -> None:
        from pathlib import Path

        _, store, storage = context()
        articles = eligible(await store.all())
        if not articles:
            typer.echo("no labelled articles — run scrape -> clean -> label first")
            raise typer.Exit(code=1)

        splits: dict[Split, list[dict[str, Any]]] = {"train": [], "eval": []}
        assigned: list[TrainingArticle] = []
        for article in articles:
            split = split_for(article.url, eval_frac, seed)
            splits[split].append(to_example(article))
            assigned.append(article.merged(split=split))
        await store.upsert_many(assigned)

        local = Path(out_dir) / version
        local.mkdir(parents=True, exist_ok=True)

        for split, rows in splits.items():
            body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
            path = local / f"{split}.jsonl"
            path.write_text(body, encoding="utf-8")
            if upload:
                await put_text(
                    storage,
                    dataset_key(version, f"{split}.jsonl"),
                    body,
                    content_type="application/jsonl",
                )
            typer.echo(f"  {split}: {len(rows)} examples -> {path}")

        card = render_data_card(version, assigned, splits, eval_frac, seed)
        (local / "data_card.md").write_text(card, encoding="utf-8")
        if upload:
            await put_text(
                storage, dataset_key(version, "data_card.md"), card, content_type="text/markdown"
            )
        typer.echo(f"  data card -> {local / 'data_card.md'}")
        typer.echo(
            f"\nnext: modal run modal/train.py --dataset-uri r2://{dataset_key(version, 'train.jsonl')}"
        )

    run_async(_run)


@app.command()
def validate(
    version: str = "v1",
    out_dir: str = ".data/datasets",
    max_chars: int = typer.Option(14000, help="Rough proxy for the 4096-token limit."),
) -> None:
    """Catch dataset defects before a training run burns GPU time."""

    async def _run() -> None:
        from pathlib import Path

        local = Path(out_dir) / version
        problems: list[str] = []
        urls: dict[str, set[str]] = {"train": set(), "eval": set()}

        for split in ("train", "eval"):
            path = local / f"{split}.jsonl"
            if not path.exists():
                problems.append(f"{split}.jsonl is missing — run `build_dataset run` first")
                continue

            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                where = f"{split}.jsonl:{lineno}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    problems.append(f"{where}: invalid JSON ({exc})")
                    continue

                messages = row.get("messages") or []
                roles = [m.get("role") for m in messages]
                if roles != ["system", "user", "assistant"]:
                    problems.append(f"{where}: roles are {roles}, expected system/user/assistant")
                    continue
                if not messages[2].get("content", "").strip():
                    problems.append(f"{where}: empty assistant turn")

                # The trained output must be parseable by the code that will
                # read it in production.
                title, meta, body = parse_article(messages[2]["content"])
                if not title or title == "Untitled":
                    problems.append(f"{where}: assistant turn has no TITLE")
                if not meta:
                    problems.append(f"{where}: assistant turn has no META")
                if body.count("\n## ") < 1:
                    problems.append(f"{where}: body has no '##' sections")

                total = sum(len(m.get("content", "")) for m in messages)
                if total > max_chars:
                    problems.append(f"{where}: {total} chars may exceed max_seq_len")

                urls[split].add((row.get("meta") or {}).get("url", where))

        if overlap := urls["train"] & urls["eval"]:
            problems.append(
                f"LEAKAGE: {len(overlap)} url(s) in both splits, e.g. {sorted(overlap)[:3]}"
            )

        typer.echo(f"train={len(urls['train'])}  eval={len(urls['eval'])}")
        if problems:
            for problem in problems[:40]:
                typer.echo(f"  ✗ {problem}")
            if len(problems) > 40:
                typer.echo(f"  … and {len(problems) - 40} more")
            raise typer.Exit(code=1)
        typer.echo("  ✓ dataset looks good")

    run_async(_run)


def render_data_card(
    version: str,
    articles: list[TrainingArticle],
    splits: dict[Split, list[dict[str, Any]]],
    eval_frac: float,
    seed: int,
) -> str:
    """Fill docs/data_card.md's template with the real numbers."""
    import statistics
    from datetime import UTC, datetime

    counts = [a.word_count for a in articles] or [0]
    by_category: dict[str, int] = {}
    by_industry: dict[str, int] = {}
    for article in articles:
        key = (article.labels.category if article.labels else None) or "unknown"
        by_category[key] = by_category.get(key, 0) + 1
        industry = (article.labels.industry if article.labels else None) or "(general)"
        by_industry[industry] = by_industry.get(industry, 0) + 1

    def table(data: dict[str, int]) -> str:
        return "\n".join(
            f"| {k} | {v} | {v / len(articles):.0%} |"
            for k, v in sorted(data.items(), key=lambda kv: -kv[1])
        )

    return f"""# Data Card — jenosize-ideas-{version}

Generated by `pipeline/build_dataset.py` on {datetime.now(UTC):%Y-%m-%d}.

## Summary

| Field | Value |
|---|---|
| Name | `jenosize-ideas-{version}` |
| Task | Conditional article generation (brief → Jenosize-style article) |
| Size | {len(articles)} examples ({len(splits["train"])} train / {len(splits["eval"])} eval) |
| Split | deterministic by sha256(url), eval_frac={eval_frac}, seed={seed} |
| Format | JSONL, OpenAI chat messages |
| Language | English |

## Provenance

Public articles from Jenosize Ideas (`jenosize.com`), discovered via sitemap and
crawled at ~1 req/s with robots.txt honoured. Raw HTML is archived to
`raw/scrape/{{date}}/{{sha256}}.html` before any extraction.

## Construction

1. `pipeline/scrape.py` — sitemap discovery, polite crawl, raw archive.
2. `pipeline/clean.py` — heading remap to `##`/`###`, boilerplate and CTA
   removal, duplicate-title removal, {{300..4000}}-word filter, shingle-Jaccard
   near-duplicate detection.
3. `pipeline/label.py` — reverse-labelled briefs; category/language/length are
   derived, not inferred; inferred fields pass through the API's normalizers.
4. `pipeline/build_dataset.py` — rendered through `app/services/prompt.py`, the
   same functions the API calls at inference.

## Statistics

| Metric | Value |
|---|---|
| Word count (min / median / mean / max) | {min(counts)} / {int(statistics.median(counts))} / {int(statistics.mean(counts))} / {max(counts)} |
| Total words | {sum(counts):,} |

### By category

| Category | Count | Share |
|---|---|---|
{table(by_category)}

### By inferred industry

| Industry | Count | Share |
|---|---|---|
{table(by_industry)}

## Known limitations

* **Single publisher.** The model learns one house voice and will not generalise
  to another brand's tone — which is the goal, but it bounds the claim.
* **Inferred briefs.** Topic, industry, audience and keywords are reconstructed
  from the finished article, so a label can be a plausible-but-wrong reading.
  `pipeline/label.py audit` samples them for human review.
* **Small corpus.** A few hundred articles teaches style, not facts; this is why
  retrieval, not the adapter, supplies evidence at inference time.
* **No source-grounded examples.** Corpus articles were not written against
  supplied sources, so the training set contains no "Reference material" blocks.
  The model's use of retrieved chunks is therefore carried by the base model and
  the system prompt, not by the fine-tune — a known gap to evaluate.
* **Recency.** Frozen at collection time; trend articles age.

## Ethics

No personal data. Author bylines and dates are stripped so the model cannot
attribute text to a real person. The house call-to-action is removed so the model
does not learn to advertise. Only the LoRA adapter is published, never the corpus.
"""


if __name__ == "__main__":
    app()
