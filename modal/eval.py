"""Base vs fine-tuned evaluation.

Run:  modal run modal/eval.py --endpoint https://...modal.run/v1 --run-name day2-r16

STATUS: skeleton. It defines *what* is measured; the scoring bodies are TODO.

Why these metrics: the deterministic ones (`app.services.quality`) are cheap and
reproducible, and a pairwise LLM judge is the only practical way to score
"sounds like Jenosize". Reporting both keeps a style win from hiding an SEO
regression.
"""

from __future__ import annotations

import json
import os

from common import BASE_MODEL, MINUTES, app, eval_image, vllm_secret

# Held-out briefs. Day 1 writes the real file to R2 as datasets/v1/eval.jsonl;
# these three exist so the harness is runnable before that lands.
FALLBACK_BRIEFS = [
    {"topic": "Agentic AI in Southeast Asian retail", "industry": "Retail & E-commerce"},
    {"topic": "Tokenised real-world assets in Thai banking", "industry": "Financial Services"},
    {
        "topic": "Circular supply chains after the tariff shock",
        "industry": "Logistics & Supply Chain",
    },
]


@app.function(image=eval_image, secrets=[vllm_secret], timeout=60 * MINUTES)
def evaluate(endpoint: str, run_name: str, adapter_name: str = "jeno-lora") -> dict:
    """Generate the same briefs from the base model and the adapter, then score."""
    from openai import OpenAI

    client = OpenAI(base_url=endpoint, api_key=os.environ.get("VLLM_API_KEY", "not-needed"))

    results: list[dict] = []
    for brief in FALLBACK_BRIEFS:
        for label, model in (("base", BASE_MODEL), ("finetuned", adapter_name)):
            # TODO(day-2): build the prompt with app.services.prompt so the eval
            # measures the *shipped* prompt, not an approximation of it.
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"Write an article about {brief['topic']}."}],
                max_tokens=2048,
            )
            results.append(
                {
                    "model_label": label,
                    "prompt": brief,
                    "output_md": completion.choices[0].message.content,
                    # TODO(day-2): deterministic scores via app.services.quality
                    # (word count, H2 count, keyword coverage, title length).
                    "scores": {},
                }
            )

    # TODO(day-2): pairwise LLM-judge on style adherence, then aggregate a
    # win-rate per dimension into `summary`.
    summary = {"run_name": run_name, "n": len(results), "judge": "TODO"}

    # TODO(day-2): write eval/{run_id}/results.json to R2 and insert into the
    # eval_runs / eval_results tables (see supabase/migrations/0001_init.sql).
    print(json.dumps(summary, indent=2))
    return {"summary": summary, "results": results}


@app.local_entrypoint()
def main(endpoint: str, run_name: str = "adhoc") -> None:
    evaluate.remote(endpoint=endpoint, run_name=run_name)
