"""Shared Modal app, volume, secret and image definitions.

Everything GPU-shaped lives in this directory and *only* here. The Vercel bundle
must never see torch/vllm/unsloth (500 MB limit), so these images are declared
inside Modal image objects — their `pip_install` lists are resolved remotely and
never touch the root `requirements.txt`.

NOTE: this directory is named `modal/`, which shadows the installed `modal`
package whenever the repo root is on `sys.path`. Running `modal deploy
modal/serve.py` or `python modal/train.py` is safe (Python puts the *script's*
directory first, not the cwd). Avoid `python -c "import modal"` from the repo
root.
"""

from __future__ import annotations

import modal

APP_NAME = "jenosize-trend-writer"

# Where the LoRA adapter and any cached weights live between runs.
MODELS_VOLUME_NAME = "jeno-models"
ADAPTER_DIR = "/models/jeno-lora-v1"

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

# Modal secrets (create with `modal secret create ...`):
#   jeno-hf    -> HF_TOKEN        (pull the base model, push the adapter)
#   jeno-vllm  -> VLLM_API_KEY    (bearer token the FastAPI backend sends)
HF_SECRET_NAME = "jeno-hf"
VLLM_SECRET_NAME = "jeno-vllm"

MINUTES = 60

app = modal.App(APP_NAME)

models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME, create_if_missing=True)
# Separate cache volume so re-downloading 8 GB of base weights is a one-time cost.
hf_cache_volume = modal.Volume.from_name("jeno-hf-cache", create_if_missing=True)

hf_secret = modal.Secret.from_name(HF_SECRET_NAME)
vllm_secret = modal.Secret.from_name(VLLM_SECRET_NAME)

VOLUMES = {
    "/models": models_volume,
    "/root/.cache/huggingface": hf_cache_volume,
}

# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
# Training and serving are split because Unsloth and vLLM pin conflicting
# torch/transformers versions; a single image would force a compromise on both.

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "unsloth==2025.9.1",
        "trl>=0.12.0",
        "peft>=0.14.0",
        "transformers>=4.51.0",
        "datasets>=3.0.0",
        "accelerate>=1.0.0",
        "bitsandbytes>=0.45.0",
        "huggingface_hub>=0.26.0",
        "boto3>=1.35.0",  # pull datasets/v1/train.jsonl straight from R2
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

serve_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.11.0", "huggingface_hub>=0.26.0")
    .env(
        {
            # Keeps cold starts honest: without this vLLM spends minutes on
            # CUDA-graph capture that a scale-to-zero service never amortises.
            "VLLM_USE_V1": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)

eval_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "openai>=1.54.0", "boto3>=1.35.0", "datasets>=3.0.0"
)
