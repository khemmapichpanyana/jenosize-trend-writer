# Architecture

## Overview

```
User / client ──► Vercel: FastAPI backend (trend-writer.workser.app)
                     │  /api/v1/*   API
                     │  /docs       OpenAPI
                     ├─► Supabase Postgres   (records; no user auth)
                     ├─► Cloudflare R2       (files, S3-compatible via boto3)
                     └─► Modal: vLLM OpenAI-compatible server
                                (Qwen3-4B-Instruct-2507 + LoRA adapter)
                                ▲
Modal: train.py (Unsloth LoRA) ─┘ writes the adapter to a Modal Volume

Cloudflare DNS: trend-writer.workser.app → CNAME → Vercel (DNS only, not proxied)
```

The browser demo page (`public/index.html`) is intentionally **not** part of this
deployment yet — this repo is the Python service.

## Responsibilities

| Component | Does | Does not |
|---|---|---|
| **Vercel** (team `workserai`) | Validation, ingestion, retrieval, prompt building, quality checks, persistence, SSE streaming. CPU only. | Any model inference, any ML dependency. |
| **Modal** | All GPU work: `train.py` (fine-tune), `serve.py` (vLLM + LoRA, scale to zero), `eval.py`. | Business logic, request validation. |
| **Supabase** | Postgres records. RLS on, no policies, service key only. | User auth — the product has none. |
| **Cloudflare R2** | Raw scraped HTML, uploads, datasets, generated markdown, eval output. | Public reads; the bucket is private. |

## Core design principle

**Fine-tuning teaches style; retrieval supplies facts.**

The adapter is never asked to remember a statistic. It learns tone, structure and
the output contract. Anything factual has to arrive in the prompt as a retrieved
chunk, which is why `retrieve.py` exists and why the system prompt explicitly
forbids inventing figures when no sources were supplied.

The practical consequence: a small (4B) base model is enough, LoRA converges in
minutes on one L4, and correctness failures are fixable by improving retrieval
rather than by retraining.

## Request flow (`POST /api/v1/articles`)

```
A. normalize        app/services/normalize.py   clean + canonicalise the brief
B. record           app/db/                     insert generations row (queued)
C. ground           ingest.py + retrieve.py     fetch/extract sources, BM25 top-k
D. generate         prompt.py + llm.py          build messages, call the model (running)
E. check            quality.py                  4 deterministic checks; 1 retry with feedback
F. persist          storage/ + db/              write markdown to R2, close the row
```

`POST /api/v1/articles/stream` runs the same flow and emits SSE events
(`status | token | heartbeat | result | error`). The heartbeat every ~10 s keeps
proxies from dropping the connection while a scaled-to-zero GPU cold-starts.

## Pluggable backends

Three seams, each a `Protocol` with dependency injection, selected by env var:

| Seam | Protocol | Implementations |
|---|---|---|
| Model | `app/services/llm.py::LLMProvider` | `MockProvider`, `OpenAICompatibleProvider` |
| Persistence | `app/db/base.py::Repository` | `NullRepository`, `SupabaseRepository` |
| Storage | `app/storage/base.py::Storage` | `LocalStorage`, `R2Storage` |

This is what makes `MODEL_PROVIDER=mock PERSISTENCE=none STORAGE=local` a real
end-to-end run rather than a stubbed one — the same orchestration code path is
exercised in CI, on a laptop and in production.

## Constraints that shaped the design

* **Vercel's 500 MB Python bundle.** No torch/transformers/vllm/unsloth in
  `requirements.txt`; heavy deps are declared only inside Modal images. CI
  enforces this with a grep.
* **Serverless has no background workers.** Generation happens inside the
  request (hence `maxDuration` and SSE) rather than in a queue.
* **Scale-to-zero GPU.** Cold starts are 1-2 minutes, so there is an explicit
  `POST /model/warmup` and an SSE heartbeat.
* **No user auth.** RLS denies everything; only the server-side key gets in. An
  optional `X-API-Key` guards the write routes against public abuse.

## Observability

Structured JSON logs to stdout (Vercel's log drain), with a request id from the
`X-Request-ID` header or generated per request, carried in a `ContextVar` and
echoed back on the response. One error envelope everywhere:
`{"error": {"code", "message", "request_id"}}`.
