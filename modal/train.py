"""LoRA fine-tune of Qwen3-4B on the Jenosize corpus (Unsloth + TRL).

    modal run modal/train.py --dataset-uri r2://datasets/v1/train.jsonl
    modal run modal/train.py --dataset-uri /models/datasets/v1/train.jsonl --push-to-hub

Why LoRA rather than a full fine-tune: the goal is *style*, not new knowledge.
A rank-16 adapter on a 4B base converges in minutes on one L4, costs a couple of
dollars per run, and ships as a ~60 MB artifact that vLLM hot-swaps without
touching the base weights — which is also what makes an honest base-vs-finetuned
comparison possible at eval time.

Why 4-bit: a 4B model in bf16 plus optimiser state does not leave room on a
24 GB L4 for a 4096-token context. QLoRA does, at a negligible quality cost for
a style adapter.
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
    r2_secret,
    train_image,
)

# --- hyperparameters --------------------------------------------------------
# Defaults are the standard QLoRA style-transfer recipe. `modal/eval.py` is what
# justifies changing any of them; do not tune them by vibes.
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LEARNING_RATE = 2e-4
EPOCHS = 3
MAX_SEQ_LEN = 4096
BATCH_SIZE = 2
GRAD_ACCUM = 4  # effective batch size 8
WARMUP_RATIO = 0.03
SEED = 3407

# All attention + MLP projections. Restricting to q/v is cheaper but noticeably
# weaker at style transfer, which is the entire point of this run.
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Qwen3 uses ChatML. These markers tell TRL where the prompt ends and the
# response begins, so loss is computed on the article only — without this the
# model spends capacity learning to reproduce briefs.
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"

HF_REPO = os.environ.get("JENO_HF_REPO", "jenosize/jeno-trend-writer-lora")
CHECKPOINT_DIR = "/models/checkpoints"


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
    secrets=[hf_secret, r2_secret],
    timeout=180 * MINUTES,
)
def train(
    dataset_uri: str = "r2://datasets/v1/train.jsonl",
    push_to_hub: bool = False,
    epochs: int = EPOCHS,
    lora_r: int = LORA_R,
    learning_rate: float = LEARNING_RATE,
    adapter_dir: str = ADAPTER_DIR,
) -> dict:
    # isort: off
    # Unsloth patches transformers/trl at import time, so it MUST be imported
    # first. Import sorting is disabled here for exactly that reason.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    # isort: on

    rows = _load_jsonl(dataset_uri)
    if not rows:
        raise ValueError(f"no training examples at {dataset_uri}")
    print(
        f"[jeno] base={BASE_MODEL} examples={len(rows)} source={dataset_uri}\n"
        f"[jeno] r={lora_r} alpha={LORA_ALPHA} lr={learning_rate} epochs={epochs} "
        f"max_seq_len={MAX_SEQ_LEN} batch={BATCH_SIZE}x{GRAD_ACCUM}"
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,  # let Unsloth pick bf16 on Ada/Ampere
        token=os.environ.get("HF_TOKEN"),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    def to_text(batch: dict) -> dict:
        # The rows already hold the exact system/user/assistant turns the API
        # sends at inference (built by pipeline/build_dataset.py through
        # app/services/prompt.py). All that is left is the chat template.
        return {
            "text": [
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                for messages in batch["messages"]
            ]
        }

    dataset = Dataset.from_list(rows).map(to_text, batched=True, remove_columns=["messages"])

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=CHECKPOINT_DIR,
            dataset_text_field="text",
            max_length=MAX_SEQ_LEN,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            warmup_ratio=WARMUP_RATIO,
            lr_scheduler_type="linear",
            optim="adamw_8bit",
            logging_steps=1,
            save_strategy="no",  # the adapter is the artifact, not the checkpoints
            seed=SEED,
            report_to="none",
            bf16=True,
        ),
    )

    # Mask the prompt: loss on the assistant turn only.
    trainer = train_on_responses_only(
        trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART
    )

    result = trainer.train()
    metrics = {
        "train_loss": float(result.training_loss),
        "steps": int(result.global_step),
        "examples": len(rows),
        "epochs": epochs,
        "lora_r": lora_r,
        "learning_rate": learning_rate,
    }
    print(f"[jeno] {metrics}")

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    with open(f"{adapter_dir}/jeno_train_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    # Commit makes the adapter visible to the serving container on its next start.
    models_volume.commit()
    print(f"[jeno] adapter written to {adapter_dir} and committed to the volume")

    if push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(HF_REPO, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=adapter_dir, repo_id=HF_REPO, repo_type="model")
        print(f"[jeno] pushed to https://huggingface.co/{HF_REPO}")

    return metrics


@app.local_entrypoint()
def train_main(
    version: str = "v1",
    dataset_uri: str | None = None,
    push_to_hub: bool = False,
    epochs: int = EPOCHS,
    lora_r: int = LORA_R,
    learning_rate: float = LEARNING_RATE,
) -> None:
    """`modal run modal/train.py --version v1`

    The version picks both the dataset (`datasets/{version}/train.jsonl`) and the
    adapter directory (`/models/jeno-lora-{version}`), so training v2 can never
    overwrite the adapter trained on v1.
    """
    metrics = train.remote(
        dataset_uri=dataset_uri or f"r2://datasets/{version}/train.jsonl",
        push_to_hub=push_to_hub,
        epochs=epochs,
        lora_r=lora_r,
        learning_rate=learning_rate,
        adapter_dir=f"/models/jeno-lora-{version}",
    )
    print(json.dumps(metrics, indent=2))
