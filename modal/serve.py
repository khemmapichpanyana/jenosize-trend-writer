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
    BASE_MODEL,
    MINUTES,
    VOLUMES,
    app,
    serve_image,
    vllm_secret,
)

VLLM_PORT = 8777


# 8192 covers the longest brief (retrieved chunks) plus a 1500-word article with
# room to spare; raising it costs KV-cache memory on a 24 GB L4.
# 8192 is useful for long retrieval contexts, but it also increases the number
# of CUDA-graph shapes compiled during a cold start. Override this for a
# throughput-focused deployment; the serverless demo defaults to eager mode
# below so the first request is not blocked by graph compilation.
MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192"))
MAX_LORA_RANK = 64  # must be >= the largest lora_r TrainParams accepts


@app.server(
    image=serve_image,
    gpu="L4",
    volumes=VOLUMES,
    secrets=[vllm_secret],
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
            # The studio agent orchestrates with tool calls on this server; vLLM
            # rejects `tools` unless auto tool choice + a parser are enabled.
            # Qwen models emit Hermes-style tool calls.
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
        ]

        # vLLM's default CUDA-graph compilation is excellent for a warm,
        # continuously running GPU, but the compile phase is repeated after a
        # scale-to-zero cold start. Eager execution removes that ~minute-long
        # startup tax. Set VLLM_ENFORCE_EAGER=0 when warm throughput matters
        # more than serverless first-request latency.
        if os.environ.get("VLLM_ENFORCE_EAGER", "1").lower() not in {"0", "false", "no"}:
            cmd.append("--enforce-eager")

        # Every complete adapter on the volume is registered under its own name
        # (jeno-lora-v1, jeno-lora-v2, ...), plus `jeno-lora` for the active one
        # (POST /v1/adapters/{v}/activate). Eval can then compare any version
        # against the base model on the same server, and the product API just
        # asks for `jeno-lora`. With no adapter yet, the base model is served.
        from pipeline.adapters import read_active, vllm_lora_args

        lora_args = vllm_lora_args("/models")
        if lora_args:
            # vLLM's default --max-lora-rank is 16; the jobs API allows r up to
            # 64, and an adapter above the limit fails to load at start-up.
            cmd += ["--enable-lora", "--max-lora-rank", str(MAX_LORA_RANK), *lora_args]
            print(f"[jeno] LoRA modules: {lora_args[1:]} (active: {read_active('/models')})")
        else:
            print(
                f"[jeno] WARNING: no trained adapter under /models; serving {BASE_MODEL} only. "
                f"Set MODEL_NAME={BASE_MODEL} on the backend until training lands."
            )

        subprocess.Popen(cmd)


@app.local_entrypoint()
def serve_main() -> None:
    """`modal run modal/serve.py` -> print the URL to paste into MODEL_BASE_URL."""
    print(f"{VLLMServer.get_url()}/v1")
