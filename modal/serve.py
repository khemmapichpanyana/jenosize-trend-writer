"""vLLM OpenAI-compatible server with the Jenosize LoRA adapter.

Deploy:   modal deploy modal/serve.py
Result:   https://<workspace>--jenosize-trend-writer-vllmserver.modal.run
Point the backend at that URL + "/v1" via MODEL_BASE_URL.

Cost shape: `min_containers=0` means the GPU is only billed while generating,
at the price of a 1-2 minute cold start. The backend's `POST /model/warmup`
exists to pay that cost while the user is still typing. Set
JENO_MIN_CONTAINERS=1 during a demo to remove cold starts entirely.
"""

from __future__ import annotations

import os
import subprocess

import modal

from common import (
    ADAPTER_DIR,
    BASE_MODEL,
    MINUTES,
    VOLUMES,
    app,
    hf_secret,
    serve_image,
    vllm_secret,
)

VLLM_PORT = 8000
LORA_NAME = "jeno-lora"

# 8192 covers the longest brief (retrieved chunks) plus a 1500-word article with
# room to spare; raising it costs KV-cache memory on a 24 GB L4.
MAX_MODEL_LEN = 8192


@app.server(
    image=serve_image,
    gpu="L4",
    volumes=VOLUMES,
    secrets=[hf_secret, vllm_secret],
    port=VLLM_PORT,
    # The API key is checked by vLLM itself, so Modal's proxy must let the
    # request through to it.
    unauthenticated=True,
    scaledown_window=5 * MINUTES,
    startup_timeout=10 * MINUTES,
    min_containers=int(os.environ.get("JENO_MIN_CONTAINERS", "0")),
)
class VLLMServer:
    @modal.enter()
    def start(self) -> None:
        cmd = [
            "vllm",
            "serve",
            BASE_MODEL,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--max-model-len",
            str(MAX_MODEL_LEN),
            "--api-key",
            os.environ.get("VLLM_API_KEY", "not-needed"),
        ]

        # The adapter only exists after `modal run modal/train.py` has committed
        # it. Serving the base model instead of crashing keeps the endpoint
        # usable on Day 0 and makes the base-vs-finetuned eval trivial.
        if os.path.isdir(ADAPTER_DIR):
            cmd += ["--enable-lora", "--lora-modules", f"{LORA_NAME}={ADAPTER_DIR}"]
            print(f"[jeno] serving {BASE_MODEL} with LoRA '{LORA_NAME}' from {ADAPTER_DIR}")
        else:
            print(
                f"[jeno] WARNING: no adapter at {ADAPTER_DIR}; serving base model only. "
                f"Set MODEL_NAME={BASE_MODEL} on the backend until training lands."
            )

        subprocess.Popen(cmd)


@app.local_entrypoint()
def main() -> None:
    """`modal run modal/serve.py` -> print the URL to paste into MODEL_BASE_URL."""
    print(f"{VLLMServer.get_url()}/v1")
