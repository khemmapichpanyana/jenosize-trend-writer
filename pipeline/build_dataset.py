"""Step 4 — publish a versioned chat-JSONL dataset to R2 for `modal/train.py`.

    uv run python -m pipeline.build_dataset run --version v1
    uv run python -m pipeline.build_dataset validate --version v1

Dataset versions are IMMUTABLE. A model card that says "trained on v1" must mean
the same bytes forever, so:
  * same corpus, same version  -> nothing is written ("unchanged");
  * changed corpus, same version -> refused; publish v2 instead
    (`--overwrite` exists for iterating *before* anything was trained on it);
  * rows are validated in memory first, and a broken dataset is never uploaded.

CRITICAL INVARIANT — prompt parity
----------------------------------
The training prompt is produced by `app.services.prompt.build_system_prompt` and
`build_user_prompt`: the *same functions the API calls at inference time*. If
this file grew its own template, the adapter would be optimised for a prompt the
service never sends. `tests/test_pipeline_dataset.py` asserts the parity.

Output, one JSON object per line:

    {"messages": [
        {"role": "system",    "content": "<style rules + output contract>"},
        {"role": "user",      "content": "<rendered brief>"},
        {"role": "assistant", "content": "TITLE: …\\nMETA: …\\n---\\n<markdown>"}
    ], "meta": {"url": …, "word_count": …}}
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import typer

from app.schemas.articles import TARGET_WORDS, NormalizedParams
from app.services import prompt as prompt_service
from app.services import quality
from app.services.llm import parse_article
from app.storage.base import Storage, put_text
from app.storage.keys import dataset_key
from pipeline._cli import echo_stats, pipeline_context, recorded_run, run_async
from pipeline.schemas import Split, TrainingArticle
from pipeline.store import CorpusStore, split_for

app = typer.Typer(help="Publish versioned train/eval JSONL to R2.")

# A meta description is part of the output contract, so a missing one has to be
# synthesised rather than left blank — otherwise the model learns to skip it.
META_FALLBACK_CHARS = 155
# Rough character proxy for MAX_SEQ_LEN=4096 tokens in modal/train.py.
MAX_EXAMPLE_CHARS = 14000
TRUNCATION_MARKER = "\n\n[Article truncated to fit the training context window.]\n"

# The normal dataset builder remains backwards-compatible with v1.  A curated
# v2 can opt into these checks so the adapter is not trained on examples that
# contradict the production output contract.  Eval rows are never removed by
# this filter: keeping the URL-hash split intact makes v1/v2 comparisons fair.
META_MIN_CHARS, META_MAX_CHARS = 120, 160
CURATED_MIN_HEADINGS = 3


class DatasetError(RuntimeError):
    """Raised instead of publishing a dataset that is invalid or would mutate a version."""


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


def _fit_article_body(
    *,
    system_prompt: str,
    user_prompt: str,
    title: str,
    meta: str,
    body: str,
) -> str:
    """Keep one rendered example inside the trainer's context-size proxy.

    The source article is the only part we shorten. We prefer a section
    boundary so the model does not learn a sentence fragment, while retaining
    the title, metadata, and prompt exactly as they are sent at inference.
    """
    assistant_prefix = f"TITLE: {title}\nMETA: {meta}\n---\n"
    available = (
        MAX_EXAMPLE_CHARS - len(system_prompt) - len(user_prompt) - len(assistant_prefix) - 1
    )
    body = body.strip()
    if len(body) <= available:
        return body

    body_budget = max(1, available - len(TRUNCATION_MARKER))
    first_heading = body.find("## ")
    if first_heading > body_budget:
        # Keep the first section heading when a long introduction would
        # otherwise consume the entire budget.
        shortened = body[first_heading : first_heading + body_budget].rstrip()
    else:
        boundary = body.rfind("\n## ", 0, body_budget)
        # Keep at least the first heading. If the only heading is the last
        # boundary found, cutting immediately before it would make the row
        # fail the production parser's section check.
        cut = boundary if boundary > first_heading else body_budget
        shortened = body[:cut].rstrip()
    return shortened + TRUNCATION_MARKER.rstrip()


def to_example(article: TrainingArticle) -> dict[str, Any]:
    """Render one article into a chat-format training example."""
    params = params_for(article)
    system_prompt = prompt_service.build_system_prompt(params)
    user_prompt = prompt_service.build_user_prompt(params, [])
    meta = meta_for(article)
    body = _fit_article_body(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        title=article.title or "",
        meta=meta,
        body=article.clean_markdown or "",
    )
    # No retrieved chunks: the corpus articles were not written against sources,
    # so training on a fabricated "Reference material" block would teach the
    # model to ignore the one we *do* send at inference time.
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {
                "role": "assistant",
                "content": (f"TITLE: {article.title}\nMETA: {meta}\n---\n{body}\n"),
            },
        ],
        # Carried for traceability; the trainer reads `messages` only.
        "meta": {"url": article.url, "word_count": article.word_count},
    }


def eligible(articles: list[TrainingArticle]) -> list[TrainingArticle]:
    return [a for a in articles if a.labels and a.clean_markdown and a.title and not a.is_duplicate]


def training_quality_issues(article: TrainingArticle) -> list[str]:
    """Return contract violations for one assistant target.

    This is deliberately separate from ``eligible``: v1 is immutable and was
    built with the original permissive rules.  New versions can opt into the
    stricter curation pass without changing old bytes or the held-out URLs.
    """
    if not article.labels or not article.clean_markdown or not article.title:
        return ["missing required training fields"]

    example = to_example(article)
    title, meta, body = parse_article(example["messages"][2]["content"])
    params = params_for(article)
    issues: list[str] = []

    if not quality.TITLE_MIN <= len(title.strip()) <= quality.TITLE_MAX:
        issues.append("title length")
    if not META_MIN_CHARS <= len(meta.strip()) <= META_MAX_CHARS:
        issues.append("meta length")
    if quality.count_h2(body) < CURATED_MIN_HEADINGS:
        issues.append("fewer than 3 H2 sections")
    floor = int(params.target_words * (1 - quality.WORD_COUNT_TOLERANCE))
    if quality.count_words(body) < floor:
        issues.append(f"body below {floor} words")
    if quality.keyword_coverage(body, params.keywords) < quality.MIN_KEYWORD_COVERAGE:
        issues.append("low keyword coverage")
    return issues


def validate_rows(splits: dict[Split, list[dict[str, Any]]]) -> list[str]:
    """Every defect that would waste a GPU run, as human-readable strings."""
    problems: list[str] = []
    urls: dict[Split, set[str]] = {"train": set(), "eval": set()}
    for split, rows in splits.items():
        for index, row in enumerate(rows, start=1):
            where = f"{split}.jsonl:{index}"
            messages = row.get("messages") or []
            roles = [m.get("role") for m in messages]
            if roles != ["system", "user", "assistant"]:
                problems.append(f"{where}: roles are {roles}, expected system/user/assistant")
                continue
            if not messages[2].get("content", "").strip():
                problems.append(f"{where}: empty assistant turn")
            # The trained output must be parseable by the code that reads it in
            # production.
            title, meta, body = parse_article(messages[2]["content"])
            if not title or title == "Untitled":
                problems.append(f"{where}: assistant turn has no TITLE")
            if not meta:
                problems.append(f"{where}: assistant turn has no META")
            if not (body.startswith("## ") or "\n## " in body):
                problems.append(f"{where}: body has no '##' sections")
            total = sum(len(m.get("content", "")) for m in messages)
            if total > MAX_EXAMPLE_CHARS:
                problems.append(f"{where}: {total} chars may exceed max_seq_len")
            urls[split].add((row.get("meta") or {}).get("url", where))
    if overlap := urls["train"] & urls["eval"]:
        problems.append(
            f"LEAKAGE: {len(overlap)} url(s) in both splits, e.g. {sorted(overlap)[:3]}"
        )
    if not splits["train"]:
        problems.append("train split is empty")
    return problems


def jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


async def run_build(
    store: CorpusStore,
    storage: Storage,
    *,
    version: str,
    eval_frac: float = 0.1,
    seed: int = 13,
    overwrite: bool = False,
    quality_filter: bool = False,
) -> dict[str, Any]:
    candidates = eligible(await store.usable())
    if not candidates:
        raise DatasetError("no labelled articles — run scrape -> clean -> label first")

    splits: dict[Split, list[dict[str, Any]]] = {"train": [], "eval": []}
    assignment: dict[str, Split] = {}
    excluded: dict[str, int] = {}
    excluded_rows = 0
    articles: list[TrainingArticle] = []
    for article in candidates:
        split = split_for(article.url, eval_frac, seed)
        # Keep every held-out URL so the v2 benchmark remains comparable to v1.
        issues = training_quality_issues(article) if quality_filter and split == "train" else []
        if issues:
            excluded_rows += 1
            for issue in issues:
                excluded[issue] = excluded.get(issue, 0) + 1
            continue
        splits[split].append(to_example(article))
        articles.append(article)
        assignment[article.url] = split

    if not splits["train"]:
        raise DatasetError("quality filter removed every training example")

    if problems := validate_rows(splits):
        raise DatasetError(
            "dataset failed validation, nothing uploaded:\n  " + "\n  ".join(problems[:40])
        )

    bodies = {split: jsonl(rows) for split, rows in splits.items()}
    digest = hashlib.sha256(f"{bodies['train']}\0{bodies['eval']}".encode()).hexdigest()
    stats: dict[str, Any] = {
        "version": version,
        "train": len(splits["train"]),
        "eval": len(splits["eval"]),
        "fingerprint": digest[:12],
        "quality_filter": quality_filter,
        "candidates": len(candidates),
        "excluded": excluded_rows,
        "excluded_reasons": excluded,
    }

    existing = await store.dataset_version(version)
    if existing and existing["fingerprint"] == digest:
        return {**stats, "status": "unchanged"}
    if existing and not overwrite:
        raise DatasetError(
            f"dataset {version} already exists with different contents "
            f"({existing['train_count']}+{existing['eval_count']} examples, created "
            f"{existing['created_at']:%Y-%m-%d}). Versions are immutable: publish a new "
            f"version (e.g. --version {_next_version(version)}), or pass --overwrite if "
            f"nothing has been trained on {version} yet."
        )

    keys = {split: dataset_key(version, f"{split}.jsonl") for split in ("train", "eval")}
    for split, body in bodies.items():
        await put_text(storage, keys[split], body, content_type="application/jsonl")
    card_key = dataset_key(version, "data_card.md")
    card = render_data_card(version, articles, splits, eval_frac, seed, digest)
    await put_text(storage, card_key, card, content_type="text/markdown; charset=utf-8")

    await store.set_splits(assignment)
    await store.record_dataset_version(
        replace=bool(existing),
        version=version,
        fingerprint=digest,
        train_count=len(splits["train"]),
        eval_count=len(splits["eval"]),
        train_key=keys["train"],
        eval_key=keys["eval"],
        card_key=card_key,
        params={"eval_frac": eval_frac, "seed": seed},
    )
    return {
        **stats,
        "status": "overwritten" if existing else "published",
        "train_key": keys["train"],
    }


def _next_version(version: str) -> str:
    if version.startswith("v") and version[1:].isdigit():
        return f"v{int(version[1:]) + 1}"
    return f"{version}-2"


@app.command()
def run(
    version: str = "v1",
    eval_frac: float = typer.Option(0.1, help="Fraction held out for evaluation."),
    seed: int = 13,
    overwrite: bool = typer.Option(False, help="Replace an existing version (only if untrained)."),
    quality_filter: bool = typer.Option(
        False,
        help="Curate training rows against the production output contract; eval URLs stay fixed.",
    ),
) -> None:
    """Build, validate and publish `datasets/{version}/` to R2 — only if it changed."""

    async def _run() -> None:
        async with pipeline_context() as (_, store, storage):
            try:
                async with recorded_run(store, "build") as stats:
                    stats.update(
                        await run_build(
                            store,
                            storage,
                            version=version,
                            eval_frac=eval_frac,
                            seed=seed,
                            overwrite=overwrite,
                            quality_filter=quality_filter,
                        )
                    )
            except DatasetError as exc:
                typer.echo(str(exc))
                raise typer.Exit(code=1) from None
            echo_stats("build", {k: v for k, v in stats.items() if k != "train_key"})
            if stats["status"] != "unchanged":
                typer.echo(f"\nnext: make train DATASET=r2://{stats['train_key']}")

    run_async(_run)


@app.command()
def validate(version: str = "v1") -> None:
    """Re-validate a published version straight from R2."""

    async def _run() -> None:
        async with pipeline_context() as (_, _store, storage):
            splits: dict[Split, list[dict[str, Any]]] = {"train": [], "eval": []}
            for split in splits:
                raw = await storage.get_bytes(dataset_key(version, f"{split}.jsonl"))
                splits[split] = [
                    json.loads(line) for line in raw.decode().splitlines() if line.strip()
                ]
        problems = validate_rows(splits)
        typer.echo(f"train={len(splits['train'])}  eval={len(splits['eval'])}")
        for problem in problems[:40]:
            typer.echo(f"  ✗ {problem}")
        if problems:
            raise typer.Exit(code=1)
        typer.echo("  ✓ dataset looks good")

    run_async(_run)


def render_data_card(
    version: str,
    articles: list[TrainingArticle],
    splits: dict[Split, list[dict[str, Any]]],
    eval_frac: float,
    seed: int,
    fingerprint: str,
) -> str:
    """Fill the data card with the real numbers for this version."""
    import statistics
    from datetime import UTC, datetime

    counts = [a.word_count for a in articles] or [0]
    by_category: dict[str, int] = {}
    by_industry: dict[str, int] = {}
    models = sorted({a.labeler_model for a in articles if a.labeler_model})
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
| Fingerprint | `{fingerprint}` (sha256 of train.jsonl + eval.jsonl) |
| Task | Conditional article generation (brief → Jenosize-style article) |
| Size | {len(articles)} examples ({len(splits["train"])} train / {len(splits["eval"])} eval) |
| Split | deterministic by sha256(url), eval_frac={eval_frac}, seed={seed} |
| Format | JSONL, OpenAI chat messages |
| Language | English |
| Labeller | {", ".join(models) or "n/a"} |

## Provenance

Public articles from Jenosize Ideas (`jenosize.com/en/ideas`), discovered via
the sitemaps and crawled at ~1 req/s with robots.txt honoured. Raw HTML is
archived in Cloudflare R2 under `raw/scrape/{{date}}/{{fingerprint}}.html`; rows
live in Postgres (`training_articles`), with every run logged in `pipeline_runs`.

## Construction

1. `pipeline/scrape.py` — sitemap discovery; incremental crawl keyed on a
   fingerprint of the extracted text (raw HTML changes per request on this site).
2. `pipeline/clean.py` — heading remap to `##`/`###`, CTA and reference-section
   removal, duplicate-title removal, 300-4000-word filter, shingle-Jaccard
   near-duplicate flagging.
3. `pipeline/label.py` — reverse-labelled briefs; category/language/length are
   derived, not inferred; inferred fields pass through the API's normalizers.
4. `pipeline/build_dataset.py` — rendered through `app/services/prompt.py`, the
   same functions the API calls at inference; validated before upload.

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
  Use of retrieved chunks is carried by the base model and the system prompt,
  not by the fine-tune — a known gap to evaluate.
* **Recency.** A snapshot at build time; trend articles age.

## Ethics

No personal data. Author bylines and dates are stripped so the model cannot
attribute text to a real person. The house call-to-action and reference lists
are removed so the model neither advertises nor fabricates citations. Neither
the corpus nor the LoRA adapter is published.
"""


if __name__ == "__main__":
    app()
