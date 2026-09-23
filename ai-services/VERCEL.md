# Vercel notes

`vercel.json` cannot carry comments, so they live here.

* **`functions["app/main.py"]`** — Vercel's Python runtime discovers a module-level
  ASGI app named `app` in this file. Nothing else is needed to route FastAPI.
* **`maxDuration: 300`** — 300 s is the Hobby ceiling. On **Pro it can go up to
  800**, which matters once `MODEL_PROVIDER=openai_compatible` points at a
  scaled-to-zero Modal GPU: a cold start plus a 1500-word generation can exceed
  five minutes. Raise it to 800 on Pro before the first real demo.
* **`rewrites`** — every path (including `/docs` and `/openapi.json`) goes to the
  single Python function. The static demo page is deliberately not part of this
  deployment yet.
* **Bundle size** — Vercel installs `requirements.txt`, not `pyproject.toml`'s
  dev/pipeline groups. Regenerate it with `make reqs` after changing runtime
  dependencies, and never add torch/transformers/vllm/unsloth: the Python bundle
  limit is 500 MB and those alone exceed it.
* **Streaming** — `POST /api/v1/articles/stream` relies on
  `X-Accel-Buffering: no` (set in `app/api/v1/articles.py`) to stop the proxy
  from buffering the SSE body.
