"""Step 4 — emit the chat-format JSONL that `modal/train.py` consumes.

    uv run python -m pipeline.build_dataset run --version v1 --eval-frac 0.1

STATUS: stub.

CRITICAL INVARIANT: the training prompt must be produced by
`app.services.prompt.build_system_prompt` / `build_user_prompt` — the very same
functions the API calls at inference time. If the dataset is built with a
hand-written prompt template, the adapter is optimised for a prompt the service
never sends, and the fine-tune silently underperforms.

Output (one JSON object per line):
    {"messages": [
        {"role": "system",    "content": "<style rules>"},
        {"role": "user",      "content": "<brief + sources>"},
        {"role": "assistant", "content": "TITLE: ...\\nMETA: ...\\n---\\n<markdown>"}
    ]}

The assistant turn reproduces the exact output contract in `prompt.py`, so the
model learns the parse target rather than being post-processed into it.
"""

from __future__ import annotations

import typer

app = typer.Typer(help="Build train/eval JSONL from labelled articles.")


@app.command()
def run(version: str = "v1", eval_frac: float = 0.1, seed: int = 13) -> None:
    """Write datasets/{version}/train.jsonl and eval.jsonl to R2 (and locally)."""
    # TODO(day-1): join training_articles + labels -> messages via app.services.prompt,
    # split deterministically by url hash (not at random, so re-runs are stable),
    # upload with app.storage.keys.dataset_key, and write the data card.
    typer.echo(f"TODO: build dataset {version} with {eval_frac:.0%} held out (seed {seed})")


@app.command()
def validate(version: str = "v1") -> None:
    """Sanity-check the JSONL before a training run burns GPU time."""
    # TODO(day-1): assert every row parses, roles alternate correctly, no empty
    # assistant turns, token length < MAX_SEQ_LEN, no train/eval URL overlap.
    typer.echo(f"TODO: validate dataset {version}")


if __name__ == "__main__":
    app()
