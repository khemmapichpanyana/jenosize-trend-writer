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
import time
import urllib.error
import urllib.request
from typing import Any

from common import (
    BASE_MODEL,
    MINUTES,
    app,
    app_image,
    pipeline_secret,
    r2_client,
    r2_secret,
    vllm_secret,
)

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


def _health_url(endpoint: str) -> str:
    """Convert an OpenAI-compatible ``.../v1`` URL to its health URL."""
    root = endpoint.rstrip("/")
    return f"{root[:-3]}/health" if root.endswith("/v1") else f"{root}/health"


def _wait_for_vllm(endpoint: str) -> None:
    """Wait through a scale-to-zero cold start before spending eval requests.

    Modal's proxy returns ``503 no upstreams available`` while the GPU
    container is booting. The OpenAI SDK retries quickly, but all 5 attempts
    can be exhausted before vLLM has loaded the base model and adapters. A
    readiness loop turns that race into bounded waiting instead of a failed
    evaluation run.
    """
    timeout_s = float(os.environ.get("EVAL_READY_TIMEOUT_S", "900"))
    poll_s = float(os.environ.get("EVAL_READY_POLL_S", "15"))
    request_timeout_s = min(float(os.environ.get("MODEL_TIMEOUT_S", "300")), 30.0)
    health = _health_url(endpoint)
    deadline = time.monotonic() + timeout_s
    last_error = "unknown readiness error"

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=request_timeout_s) as response:
                if 200 <= response.status < 300:
                    print(f"[jeno] vLLM ready: {health}")
                    return
                last_error = f"HTTP {response.status}"
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        print(f"[jeno] vLLM not ready ({last_error}); retrying in {poll_s:g}s")
        time.sleep(poll_s)

    raise RuntimeError(f"vLLM did not become ready within {timeout_s:g}s: {health} ({last_error})")


def _complete_with_retry(
    client: Any, *, model: str, messages: list[dict[str, str]], seed: int
) -> Any:
    """Retry transient Modal/vLLM failures with a long cold-start backoff."""
    from openai import APIConnectionError, APIError, APITimeoutError

    attempts = int(os.environ.get("MODEL_RETRY_ATTEMPTS", "5"))
    initial_s = float(os.environ.get("MODEL_RETRY_INITIAL_S", "10"))
    max_s = float(os.environ.get("MODEL_RETRY_MAX_S", "60"))
    for attempt in range(attempts):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                max_completion_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                seed=seed,
            )
        except (APIConnectionError, APITimeoutError, APIError) as exc:
            status = getattr(exc, "status_code", None)
            transient = status is None or status in {408, 409, 425, 429, 500, 502, 503, 504}
            if not transient or attempt == attempts - 1:
                raise
            delay = min(initial_s * (2**attempt), max_s)
            print(
                f"[jeno] transient vLLM error ({status or type(exc).__name__}); retrying in {delay:g}s"
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def _load_briefs(uri: str | None) -> list[dict[str, Any]]:
    if not uri:
        return FALLBACK_BRIEFS
    if uri.startswith("r2://"):
        client, bucket = r2_client()
        body = client.get_object(Bucket=bucket, Key=uri[5:])["Body"].read()
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
        user = messages[1]["content"]
        briefs.append(
            {
                "system": messages[0]["content"],
                "user": user,
                "reference": messages[2]["content"],
                "url": (row.get("meta") or {}).get("url"),
                # Recovered from the rendered brief so the deterministic scores
                # judge each article against *its* keywords and length, not
                # defaults (which would make keyword coverage trivially 100%).
                **_brief_fields(user),
            }
        )
    return briefs


def _brief_fields(user_prompt: str) -> dict[str, Any]:
    import re

    fields: dict[str, Any] = {}
    if match := re.search(r"^Topic: (.+)$", user_prompt, re.MULTILINE):
        fields["topic"] = match.group(1).strip()
    if match := re.search(r"^SEO keywords: (.+)$", user_prompt, re.MULTILINE):
        fields["keywords"] = [k.strip() for k in match.group(1).split(",") if k.strip()]
    if match := re.search(r"^Target length: about (\d+) words$", user_prompt, re.MULTILINE):
        words = int(match.group(1))
        fields["length"] = {600: "short", 1000: "medium", 1500: "long"}.get(words, "medium")
    return fields


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
    image=app_image,
    # jeno-pipeline carries the LABELER_* settings, reused as the default judge.
    secrets=[vllm_secret, r2_secret, pipeline_secret],
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
    judge: bool = False,
) -> dict:
    """Base vs `adapter_name` on the held-out briefs.

    With `judge=True` and no explicit judge, the labelling LLM (LABELER_*) is
    the judge. It must not be the served model: a model grading itself against
    its own fine-tune is not independent evidence.
    """
    judge_key = os.environ.get("VLLM_API_KEY", "not-needed")
    if judge and not judge_model:
        judge_model = os.environ.get("LABELER_MODEL")
        judge_endpoint = os.environ.get("LABELER_BASE_URL")
        judge_key = os.environ.get("LABELER_API_KEY") or "not-needed"
        if not (judge_model and judge_endpoint):
            raise RuntimeError(
                "judge=True needs LABELER_BASE_URL and LABELER_MODEL in jeno-pipeline"
            )
    from openai import OpenAI

    from app.services.llm import parse_article

    briefs = _load_briefs(briefs_uri)[:limit]
    _wait_for_vllm(endpoint)
    client = OpenAI(
        base_url=endpoint,
        api_key=os.environ.get("VLLM_API_KEY", "not-needed"),
        # Evaluation calls the same scale-to-zero server as production. Let the
        # SDK retry transient 502/503/504 cold-start responses, but keep the
        # per-request timeout finite so a broken endpoint still fails clearly.
        timeout=float(os.environ.get("MODEL_TIMEOUT_S", "300")),
        # Retries are explicit in _complete_with_retry so the cold-start wait
        # is visible in Modal logs and does not exhaust before vLLM is ready.
        max_retries=0,
    )
    candidates = {"base": BASE_MODEL, "finetuned": adapter_name}

    results: list[dict[str, Any]] = []
    for index, brief in enumerate(briefs):
        messages = _messages_for(brief)
        generated: dict[str, tuple[str, str]] = {}

        for label, model in candidates.items():
            completion = _complete_with_retry(client, model=model, messages=messages, seed=index)
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
                judge_key,
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
    endpoint: str,
    model: str,
    api_key: str,
    brief: str,
    output_a: str,
    output_b: str,
    *,
    seed: int,
) -> dict[str, Any]:
    """Blind pairwise comparison with randomised presentation order."""
    from openai import OpenAI

    client = OpenAI(
        base_url=endpoint,
        api_key=api_key,
        timeout=float(os.environ.get("MODEL_TIMEOUT_S", "300")),
        max_retries=int(os.environ.get("MODEL_RETRY_ATTEMPTS", "5")),
    )
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
        max_completion_tokens=200,
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

    run_id = f"{run_name}-{uuid.uuid4().hex[:8]}"
    client, bucket = r2_client()
    client.put_object(
        Bucket=bucket,
        Key=f"eval/{run_id}/results.json",
        Body=json.dumps(payload, indent=2).encode(),
        ContentType="application/json",
    )
    print(f"[jeno] results -> eval/{run_id}/results.json")


@app.local_entrypoint()
def eval_main(
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
