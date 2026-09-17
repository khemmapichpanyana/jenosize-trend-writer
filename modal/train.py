"""LoRA fine-tune of Qwen3-4B on the Jenosize corpus (Unsloth).

Run:  modal run modal/train.py --dataset-uri r2://datasets/v1/train.jsonl
      modal run modal/train.py --dataset-uri /local/path/train.jsonl --push-to-hub

STATUS: skeleton. The plumbing (image, volume, dataset loading, adapter commit,
HF push) is real and runnable; the training call itself is marked TODO and is
Day 2's work.

Why LoRA rather than a full fine-tune: the goal is *style*, not new knowledge.
A rank-16 adapter on a 4B base converges in minutes on an L4, costs a few
dollars per run, and ships as a ~60 MB artifact that vLLM can hot-swap.
"""

from __future__ import annotations

import json
import os

from common import (
    ADAPTER_DIR,
    BASE_MODEL,
    MINUTES,
    VOLUMES,
    app,
    hf_secret,
    models_volume,
    train_image,
)

# --- hyperparameters (Day 2 will tune these against modal/eval.py) -----------
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LEARNING_RATE = 2e-4
EPOCHS = 3
MAX_SEQ_LEN = 4096
BATCH_SIZE = 2
GRAD_ACCUM = 4

HF_REPO = "jenosize/jeno-trend-writer-lora"


def _load_jsonl(uri: str) -> list[dict]:
    """Load the training set from R2 (`r2://key`) or a path inside the container."""
    if uri.startswith("r2://"):
        import boto3

        key = uri[len("r2://") :]
        client = boto3.client(
            "s3",
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )
        body = client.get_object(Bucket=os.environ["R2_BUCKET"], Key=key)["Body"].read()
        lines = body.decode().splitlines()
    else:
        with open(uri) as fh:
            lines = fh.read().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@app.function(
    image=train_image,
    gpu="L4",
    volumes=VOLUMES,
    secrets=[hf_secret],
    timeout=120 * MINUTES,
)
def train(dataset_uri: str = "r2://datasets/v1/train.jsonl", push_to_hub: bool = False) -> str:
    rows = _load_jsonl(dataset_uri)
    print(
        f"[jeno] base={BASE_MODEL} examples={len(rows)} source={dataset_uri} "
        f"r={LORA_R} alpha={LORA_ALPHA} lr={LEARNING_RATE} epochs={EPOCHS} "
        f"max_seq_len={MAX_SEQ_LEN} batch={BATCH_SIZE}x{GRAD_ACCUM}"
    )

    # Each row is {"messages": [{"role": "system"|"user"|"assistant", ...}]},
    # built by pipeline/build_dataset.py so that the *training* prompt is byte
    # for byte the prompt app/services/prompt.py builds at inference time.
    # A mismatch here is the single most common cause of "the adapter did
    # nothing" — do not let these two drift apart.

    from unsloth import FastLanguageModel  # noqa: F401  (import cost is ~30 s)

    # TODO(day-2): load the base model in 4-bit.
    #   model, tokenizer = FastLanguageModel.from_pretrained(
    #       model_name=BASE_MODEL,
    #       max_seq_length=MAX_SEQ_LEN,
    #       load_in_4bit=True,
    #   )
    #
    # TODO(day-2): attach the LoRA adapter.
    #   model = FastLanguageModel.get_peft_model(
    #       model,
    #       r=LORA_R,
    #       lora_alpha=LORA_ALPHA,
    #       lora_dropout=LORA_DROPOUT,
    #       target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
    #                       "gate_proj", "up_proj", "down_proj"],
    #       use_gradient_checkpointing="unsloth",
    #   )
    #
    # TODO(day-2): train with TRL's SFTTrainer, masking the prompt so the loss
    # is computed on the assistant turn only (otherwise the model learns to
    # reproduce briefs, not articles).
    #   trainer = SFTTrainer(model=model, tokenizer=tokenizer,
    #                        train_dataset=Dataset.from_list(rows), ...)
    #   trainer.train()
    #
    # TODO(day-2): model.save_pretrained_merged? No — keep the adapter separate
    # so vLLM can serve base + LoRA and the eval can compare them.
    #   model.save_pretrained(ADAPTER_DIR)
    #   tokenizer.save_pretrained(ADAPTER_DIR)

    os.makedirs(ADAPTER_DIR, exist_ok=True)

    # Commit makes the adapter visible to the serving container on its next start.
    models_volume.commit()
    print(f"[jeno] adapter written to {ADAPTER_DIR} and committed to the volume")

    if push_to_hub:
        # TODO(day-2): publish a public adapter + model card (part of the grade).
        #   from huggingface_hub import HfApi
        #   HfApi(token=os.environ["HF_TOKEN"]).upload_folder(
        #       folder_path=ADAPTER_DIR, repo_id=HF_REPO, repo_type="model")
        print(f"[jeno] TODO: push {ADAPTER_DIR} to {HF_REPO}")

    return ADAPTER_DIR


@app.local_entrypoint()
def main(dataset_uri: str = "r2://datasets/v1/train.jsonl", push_to_hub: bool = False) -> None:
    print(train.remote(dataset_uri=dataset_uri, push_to_hub=push_to_hub))
