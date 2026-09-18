"""Base vs fine-tuned evaluation.

    modal run modal/eval.py --endpoint https://<ws>--jenosize-trend-writer-vllmserver.modal.run/v1

Two families of metric, because either alone is misleading:

  * **Deterministic** (word count, `##` count, keyword coverage, title length) —
    cheap, reproducible, and exactly the gate the live API applies. If the
    fine-tune wins on style but regresses here, the product got worse.
  * **Pairwise LLM judge** — the only practical way to score "does this sound
    like Jenosize". Run blind, with the two candidates presented in a randomised
    order so position bias does not decide the winner.

Both models are served by the *same* vLLM container: the base model by its HF
id, the fine-tune by the LoRA module name. Same weights, same sampler, same
hardware — the adapter is the only variable.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

from common import BASE_MODEL, MINUTES, app, eval_image, r2_secret, vllm_secret

# Held-out briefs. `--briefs-uri r2://datasets/v1/eval.jsonl` replaces these with
# the real held-out split; these three keep the harness runnable before then.
FALLBACK_BRIEFS: list[dict[str, Any]] = [
    {
        "topic": "Agentic AI in Southeast Asian retail",
        "category": "Futurist",
        "industry": "Retail & E-commerce",
        "audience": "C-suite executives",
        "keywords": ["agentic ai", "retail media", "personalization"],
        "length": "medium",
    },
    {
        "topic": "Tokenised real-world assets in Thai banking",
        "category": "Transformation and Technology",
        "industry": "Financial Services",
        "audience": "Banking executives",
        "keywords": ["tokenisation", "real-world assets", "compliance"],
        "length": "medium",
    },
    {
        "topic": "Circular supply chains after the tariff shock",
        "category": "Utility for Our World",
        "industry": "Logistics & Supply Chain",
        "audience": "Operations leaders",
        "keywords": ["circular economy", "supply chain resilience", "tariffs"],
        "length": "medium",
    },
]

JUDGE_SYSTEM = """You are comparing two business articles written from the same brief.

Judge only these dimensions, in this order of importance:
1. Voice: executive, concrete, forward-looking; no hype or filler.
2. Structure: a hook, then clearly-scoped sections, then an actionable close.
3. Specificity: named technologies, markets and timeframes over adjectives.
4. Restraint: no invented statistics, companies, dates or quotes.

Reply with JSON only: {"winner": "A" | "B" | "tie", "reason": "<one sentence>"}"""

TEMPERATURE = 0.7
MAX_TOKENS = 2048


def _load_briefs(uri: str | None) -> list[dict[str, Any]]:
    if not uri:
        return FALLBACK_BRIEFS
    if uri.startswith("r2://"):
        import boto3

        client = boto3.client(
            "s3",
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )
        body = client.get_object(Bucket=os.environ["R2_BUCKET"], Key=uri[5:])["Body"].read()
        lines = body.decode().splitlines()
    else:
        with open(uri) as fh:
            lines = fh.read().splitlines()

    # eval.jsonl holds full chat examples; the brief is the user turn, which we
    # replay verbatim so the eval measures the shipped prompt.
    briefs = []
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row["messages"]
        briefs.append(
            {
                "system": messages[0]["content"],
                "user": messages[1]["content"],
                "reference": messages[2]["content"],
                "url": (row.get("meta") or {}).get("url"),
            }
        )
    return briefs


def _messages_for(brief: dict[str, Any]) -> list[dict[str, str]]:
    """Build the prompt exactly as the API would.

    `app.services.prompt` is imported here rather than reimplemented so a change
    to the shipped prompt is automatically reflected in the eval.
    """
    if "user" in brief:
        return [
            {"role": "system", "content": brief["system"]},
            {"role": "user", "content": brief["user"]},
        ]

    from app.schemas.articles import TARGET_WORDS, NormalizedParams
    from app.services import prompt as prompt_service

    params = NormalizedParams(
        topic=brief["topic"],
        category=brief.get("category"),
        industry=brief.get("industry"),
        audience=brief.get("audience", "Business leaders"),
        keywords=brief.get("keywords", []),
        length=brief.get("length", "medium"),
        target_words=TARGET_WORDS[brief.get("length", "medium")],
    )
    return [
        {"role": "system", "content": prompt_service.build_system_prompt(params)},
        {"role": "user", "content": prompt_service.build_user_prompt(params, [])},
    ]


def _score(markdown: str, title: str, brief: dict[str, Any]) -> dict[str, Any]:
    """The live API's own quality gate, reused verbatim."""
    from app.schemas.articles import TARGET_WORDS, NormalizedParams
    from app.services import quality

    length = brief.get("length", "medium")
    params = NormalizedParams(
        topic=brief.get("topic", ""),
        keywords=brief.get("keywords", []),
        length=length,
        target_words=TARGET_WORDS[length],
    )
    return json.loads(quality.evaluate(markdown, title, params).model_dump_json())


@app.function(
    image=eval_image.add_local_python_source("app"),
    secrets=[vllm_secret, r2_secret],
    timeout=90 * MINUTES,
)
def evaluate(
    endpoint: str,
    run_name: str = "adhoc",
    adapter_name: str = "jeno-lora",
    briefs_uri: str | None = None,
    judge_model: str | None = None,
    judge_endpoint: str | None = None,
    limit: int = 20,
) -> dict:
    from openai import OpenAI

    from app.services.llm import parse_article

    briefs = _load_briefs(briefs_uri)[:limit]
    client = OpenAI(base_url=endpoint, api_key=os.environ.get("VLLM_API_KEY", "not-needed"))
    candidates = {"base": BASE_MODEL, "finetuned": adapter_name}

    results: list[dict[str, Any]] = []
    for index, brief in enumerate(briefs):
        messages = _messages_for(brief)
        generated: dict[str, tuple[str, str]] = {}

        for label, model in candidates.items():
            completion = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                # Fixed seed: the two candidates should differ because of the
                # adapter, not because of sampling luck.
                seed=index,
            )
            raw = completion.choices[0].message.content or ""
            title, meta, body = parse_article(raw)
            generated[label] = (title, body)
            results.append(
                {
                    "model_label": label,
                    "prompt": brief,
                    "output_md": raw,
                    "scores": _score(body, title, brief),
                    "meta_description": meta,
                }
            )

        if judge_model:
            verdict = _judge(
                judge_endpoint or endpoint,
                judge_model,
                messages[1]["content"],
                generated["base"][1],
                generated["finetuned"][1],
                seed=index,
            )
            results[-1]["judge"] = verdict

    summary = _summarise(run_name, results)
    print(json.dumps(summary, indent=2))

    if briefs_uri and briefs_uri.startswith("r2://"):
        _upload_results(run_name, {"summary": summary, "results": results})

    return {"summary": summary, "results": results}


def _judge(
    endpoint: str, model: str, brief: str, output_a: str, output_b: str, *, seed: int
) -> dict[str, Any]:
    """Blind pairwise comparison with randomised presentation order."""
    from openai import OpenAI

    client = OpenAI(base_url=endpoint, api_key=os.environ.get("VLLM_API_KEY", "not-needed"))
    flipped = random.Random(seed).random() < 0.5
    first, second = (output_b, output_a) if flipped else (output_a, output_b)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": f"BRIEF:\n{brief}\n\n=== ARTICLE A ===\n{first}\n\n=== ARTICLE B ===\n{second}",
            },
        ],
        temperature=0.0,
        max_tokens=200,
    )
    try:
        raw = completion.choices[0].message.content or "{}"
        verdict = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
    except (ValueError, IndexError):
        return {"winner": "tie", "reason": "unparseable judge reply"}

    winner = verdict.get("winner", "tie")
    if winner in ("A", "B"):
        # Undo the shuffle: report base/finetuned, not the position shown.
        position_is_base = (winner == "A") != flipped
        verdict["winner"] = "base" if position_is_base else "finetuned"
    return verdict


def _summarise(run_name: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(label: str, key: str) -> float:
        values = [r["scores"][key] for r in results if r["model_label"] == label]
        return round(sum(values) / len(values), 3) if values else 0.0

    judged = [r["judge"]["winner"] for r in results if r.get("judge")]
    return {
        "run_name": run_name,
        "n_briefs": len({id(r["prompt"]) for r in results}),
        "deterministic": {
            label: {
                "pass_rate": mean(label, "passed"),
                "keyword_coverage": mean(label, "keyword_coverage"),
                "word_count": mean(label, "word_count"),
                "heading_count": mean(label, "heading_count"),
                "title_length": mean(label, "title_length"),
            }
            for label in ("base", "finetuned")
        },
        "judge": {
            "finetuned_wins": judged.count("finetuned"),
            "base_wins": judged.count("base"),
            "ties": judged.count("tie"),
        }
        if judged
        else None,
    }


def _upload_results(run_name: str, payload: dict[str, Any]) -> None:
    import uuid

    import boto3

    run_id = f"{run_name}-{uuid.uuid4().hex[:8]}"
    boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    ).put_object(
        Bucket=os.environ["R2_BUCKET"],
        Key=f"eval/{run_id}/results.json",
        Body=json.dumps(payload, indent=2).encode(),
        ContentType="application/json",
    )
    print(f"[jeno] results -> eval/{run_id}/results.json")


@app.local_entrypoint()
def main(
    endpoint: str,
    run_name: str = "adhoc",
    adapter_name: str = "jeno-lora",
    briefs_uri: str | None = None,
    judge_model: str | None = None,
    judge_endpoint: str | None = None,
    limit: int = 20,
) -> None:
    evaluate.remote(
        endpoint=endpoint,
        run_name=run_name,
        adapter_name=adapter_name,
        briefs_uri=briefs_uri,
        judge_model=judge_model,
        judge_endpoint=judge_endpoint,
        limit=limit,
    )
