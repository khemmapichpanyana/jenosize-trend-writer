"""Shared Modal app, volume, secret and image definitions.

Everything GPU-shaped lives in this directory and *only* here. The Vercel bundle
must never see torch/vllm/unsloth (500 MB limit), so these images are declared
inside Modal image objects — their `pip_install` lists are resolved remotely and
never touch the root `requirements.txt`.

NOTE: this directory is named `modal/` but has no `__init__.py`, so it is only a
namespace-package candidate. Python prefers the real `modal` package in
site-packages, and `import modal` resolves correctly even from the repo root.
Do not add an `__init__.py` here — that would make it a regular package and
genuinely shadow the SDK.
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "jenosize-trend-writer"

# Where the LoRA adapter and any cached weights live between runs.
MODELS_VOLUME_NAME = "jeno-models"

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

# Modal secrets (create with `modal secret create ...`):
#   jeno-hf    -> HF_TOKEN        (pull the base model, push the adapter)
#   jeno-vllm  -> VLLM_API_KEY    (bearer token the FastAPI backend sends)
#   jeno-r2    -> R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY,
#                 R2_API_ENDPOINT, R2_JENOSIZE_BUCKET (same names as .env)
#   jeno-pipeline -> SUPABASE_URL, DB_PASSWORD, DB_HOST (or DATABASE_URL),
#                    JOBS_API_KEY, LABELER_* (the jobs API + workers)
# `make modal-secrets` creates both from .env, with only these keys.
HF_SECRET_NAME = "jeno-hf"
VLLM_SECRET_NAME = "jeno-vllm"
R2_SECRET_NAME = "jeno-r2"
PIPELINE_SECRET_NAME = "jeno-pipeline"

MINUTES = 60

app = modal.App(APP_NAME)

models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME, create_if_missing=True)
# Separate cache volume so re-downloading 8 GB of base weights is a one-time cost.
hf_cache_volume = modal.Volume.from_name("jeno-hf-cache", create_if_missing=True)

hf_secret = modal.Secret.from_name(HF_SECRET_NAME)
vllm_secret = modal.Secret.from_name(VLLM_SECRET_NAME)
r2_secret = modal.Secret.from_name(R2_SECRET_NAME)
pipeline_secret = modal.Secret.from_name(PIPELINE_SECRET_NAME)

VOLUMES = {
    "/models": models_volume,
    "/root/.cache/huggingface": hf_cache_volume,
}

# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
# Modal uploads only the file a function is defined in (checked in the SDK:
# `get_entrypoint_mount`). Every function here imports this `common` module at
# the top, so each image adds it explicitly — without that, containers fail at
# import with `ModuleNotFoundError: common` before doing any work.
#
# Training and serving are split because Unsloth and vLLM pin conflicting
# torch/transformers versions; a single image would force a compromise on both.

REPO_ROOT = Path(__file__).resolve().parent.parent


def r2_client():  # type: ignore[no-untyped-def]
    """(boto3 S3 client, bucket) from the `jeno-r2` secret, inside a container.

    Lives here, not in `app`, because the training image carries no app code.
    Accepts the same names as .env (R2_API_ENDPOINT / R2_JENOSIZE_BUCKET) and
    the generic ones (R2_ENDPOINT / R2_BUCKET), mirroring app.core.config.
    """
    import os
    from urllib.parse import urlparse

    import boto3

    endpoint = os.environ.get("R2_API_ENDPOINT") or os.environ.get("R2_ENDPOINT")
    if endpoint:
        parsed = urlparse(endpoint)
        endpoint = f"{parsed.scheme}://{parsed.netloc}"  # strip a pasted /<bucket>
    else:
        endpoint = f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    bucket = os.environ.get("R2_JENOSIZE_BUCKET") or os.environ["R2_BUCKET"]
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    return client, bucket


train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    # Versions are pinned loosely on purpose: Unsloth tracks transformers/trl
    # closely and pinning all four exactly is the fastest way to an unsolvable
    # resolve. If a run breaks, pin `unsloth` first and let it choose the rest.
    .uv_pip_install(
        "unsloth",
        "unsloth_zoo",
        "trl",
        "peft",
        "transformers",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "huggingface_hub",
        "hf_transfer",
        "boto3",  # read datasets/v1/train.jsonl straight from R2
        # live telemetry: per-step loss + GPU gauges written to Postgres
        "psycopg[binary]",
        "pydantic-settings",  # app.core.config resolves the DB URL the same way everywhere
        "nvidia-ml-py",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("common", "app", "pipeline")
)

serve_image = (
    modal.Image.debian_slim(python_version="3.12")
    # hf_transfer must be installed whenever HF_HUB_ENABLE_HF_TRANSFER=1, or
    # huggingface_hub refuses to download at all.
    .uv_pip_install("vllm==0.11.0", "huggingface_hub>=0.26.0", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # `pipeline.adapters` is stdlib-only; it decides which adapters to register.
    .add_local_python_source("common", "pipeline")
)


# CPU image for everything that runs this repo's own code: the jobs API, job
# workers and eval. Built from the same requirements.txt Vercel installs, plus
# the pipeline's Postgres driver, so "works locally" and "works on Modal" share
# one dependency list.
def _dependency_group(name: str) -> list[str]:
    """A [dependency-groups] list from pyproject.toml — one source of truth."""
    import tomllib

    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        return list(tomllib.load(fh)["dependency-groups"][name])


app_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(REPO_ROOT / "requirements.txt"))
    .uv_pip_install("psycopg[binary]>=3.2.0", *_dependency_group("agent"))
    .add_local_python_source("app", "pipeline", "studio", "common")
    # The agent's brand context + page theme (not .py, so added explicitly).
    .add_local_dir(str(REPO_ROOT / "studio" / "brand"), remote_path="/root/studio/brand")
    # POST /v1/migrations/apply reads these next to the pipeline package, where
    # pipeline/migrate.py expects them (<root>/supabase/migrations).
    .add_local_dir(
        str(REPO_ROOT / "supabase" / "migrations"), remote_path="/root/supabase/migrations"
    )
)
