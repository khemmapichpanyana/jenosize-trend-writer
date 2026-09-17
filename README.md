# Jenosize Trend Writer

AI service that generates business trend and future-ideas articles in the voice of
**Jenosize Ideas**. FastAPI backend + a fine-tuned Qwen3-4B LoRA served on Modal,
with BM25 retrieval over user-supplied sources.

> **Core design principle: fine-tuning teaches style, retrieval supplies facts.**

---

## Run it locally in 4 commands

No cloud accounts, no GPU, no API keys.

```bash
uv sync --group dev          # 1. install (Python 3.12)
cp .env.example .env         # 2. defaults are already the zero-account mode
make dev                     # 3. http://localhost:8000/docs
./scripts/smoke_test.sh      # 4. health + generate + stream, in another shell
```

Generate an article:

```bash
curl -s localhost:8000/api/v1/articles -H 'Content-Type: application/json' -d '{
  "topic": "Agentic AI in Southeast Asian retail",
  "category": "Futurist",
  "industry": "ecommerce",
  "audience": "C-suite executives",
  "keywords": ["agentic ai", "retail media", "personalization"],
  "length": "short"
}' | jq '{title, quality_report}'
```

Stream it (SSE):

```bash
curl -N localhost:8000/api/v1/articles/stream -H 'Content-Type: application/json' \
  -d '{"topic":"The future of embedded finance","industry":"fintech","length":"short"}'
```

`make lint test` runs ruff + mypy + the 58-test suite.

---

## Architecture

```
Client ──► Vercel: FastAPI backend (trend-writer.workser.app)
              │  /api/v1/*   API
              │  /docs       OpenAPI
              ├─► Supabase Postgres   (records; no user auth, RLS deny-all)
              ├─► Cloudflare R2       (files, S3-compatible via boto3)
              └─► Modal: vLLM OpenAI-compatible server
                         (Qwen3-4B-Instruct-2507 + LoRA adapter, scale to zero)
                         ▲
Modal: train.py (Unsloth LoRA) ─┘ adapter → Modal Volume + Hugging Face Hub
```

Vercel does CPU work only. **No ML dependency may enter `requirements.txt`** —
the Python bundle limit is 500 MB, and torch alone blows it. Heavy packages are
declared exclusively inside the Modal images in `modal/common.py`; CI greps for
violations.

Full detail: [`docs/architecture.md`](docs/architecture.md).

### Generation flow

```
A. normalize  →  B. record (queued)  →  C. ingest + retrieve  →
D. prompt + generate (running)  →  E. quality gate (one retry with feedback)  →
F. persist markdown + close the row (succeeded / failed)
```

### Pluggable backends

Every external system sits behind a `Protocol` with dependency injection, chosen
by env var. That is what makes the local mode a *real* end-to-end run:

| Seam | Env var | Options |
|---|---|---|
| Model | `MODEL_PROVIDER` | `mock` (deterministic, streams too) · `openai_compatible` (Modal vLLM) |
| Persistence | `PERSISTENCE` | `none` (in-memory) · `supabase` |
| Storage | `STORAGE` | `local` (`./.data/`) · `r2` |

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | `{status, version, checks:{database, storage, model}}` — always 200 |
| `POST` | `/api/v1/model/warmup` | Wake the scaled-to-zero GPU; `{status, latency_ms}` |
| `POST` | `/api/v1/sources/upload` | multipart pdf/docx/txt/md, max 10 MB → `{source_id, title, text_chars}` |
| `POST` | `/api/v1/articles` | `ArticleRequest` → `ArticleResponse` (blocking) |
| `POST` | `/api/v1/articles/stream` | SSE: `status \| token \| heartbeat \| result \| error` |
| `GET` | `/api/v1/articles?limit=` | Recent generations (summary rows) |
| `GET` | `/api/v1/articles/{id}` | One full generation |

Interactive docs at `/docs`. Errors are uniform:
`{"error": {"code", "message", "request_id"}}`.

**`ArticleRequest`** — `topic` (3–200 chars, required), `category`, `industry`,
`audience`, `keywords` (list *or* comma-separated string), `source_url`,
`source_ids`, `length` (`short|medium|long` → 600/1000/1500 words), `language`
(`en|th`).

If `API_KEY` is set, POST routes require the `X-API-Key` header.

---

## Repository layout

```
app/
  main.py            FastAPI factory, request-id middleware, JSON logging
  api/v1/            health · model · sources · articles (SSE)
  core/              config (pydantic-settings) · logging · errors · deps
  schemas/           Pydantic v2 request/response contracts
  services/
    normalize.py     ✅ full — whitespace, keyword dedupe/cap, industry canon
    retrieve.py      ✅ full — paragraph chunking + Okapi BM25
    quality.py       ✅ full — word count, H2 count, keyword coverage, title len
    prompt.py        ✅ system style rules + brief template (rules are placeholder)
    llm.py           ✅ LLMProvider protocol · MockProvider · OpenAICompatibleProvider
    generation.py    ✅ the A→F orchestrator, blocking + streaming
    ingest.py        ✅ URL → trafilatura · PDF/DOCX/TXT → text
  db/                Repository protocol · NullRepository · SupabaseRepository
  storage/           Storage protocol · LocalStorage · R2Storage · key conventions
pipeline/            🚧 stubs — scrape · clean · label · build_dataset (typer CLIs)
modal/               🚧 skeletons — common · train (Unsloth LoRA) · serve (vLLM) · eval
supabase/migrations/ 0001_init.sql — 7 tables, RLS enabled, no policies
tests/               58 tests: normalize · retrieve · quality · API (mock, ASGI transport)
docs/                architecture.md · report.md (outline) · data_card.md (template)
scripts/             smoke_test.sh · gen_requirements.sh
```

### Done vs stubbed

**Done and tested** — the whole API surface, the A→F generation flow, both
transports, normalization, chunking + BM25 retrieval, the quality gate with its
retry, ingestion of URLs and documents, all three seams with every
implementation, the SQL schema, structured logging, the error model, CI.

**Stubbed** — `pipeline/*` (CLI shape and docstrings final, bodies are TODO),
`modal/train.py` (plumbing runnable, the Unsloth training call is TODO),
`modal/eval.py` (metrics defined, scoring TODO). `prompt.STYLE_RULES` is written
from a skim of the corpus and gets replaced by Day 1's labelling output. The
browser demo page is not in this repo yet.

---

## Storage key conventions

```
raw/scrape/{yyyy-mm-dd}/{content_hash}.html
uploads/{source_id}/{safe_filename}
datasets/{version}/train.jsonl | eval.jsonl | data_card.md
generations/{generation_id}.md
eval/{run_id}/results.json
```

`STORAGE=local` mirrors these paths under `./.data/`, so what you inspect locally
is what production writes.

---

## Environment variables

See [`.env.example`](.env.example) for the annotated list. The ones that change
behaviour:

| Variable | Default | Notes |
|---|---|---|
| `MODEL_PROVIDER` | `mock` | `openai_compatible` needs `MODEL_BASE_URL` |
| `MODEL_BASE_URL` | — | `https://<workspace>--jenosize-trend-writer-vllmserver.modal.run/v1` |
| `MODEL_NAME` | `jeno-lora` | Use the base model id until the adapter is trained |
| `MODEL_TIMEOUT_S` | `240` | A cold L4 takes 1–2 minutes to first token |
| `PERSISTENCE` | `none` | `supabase` needs `SUPABASE_URL` + `SUPABASE_SECRET_KEY` |
| `STORAGE` | `local` | `r2` needs the three `R2_*` credentials |
| `API_KEY` | unset | When set, POST routes require `X-API-Key` |
| `CORS_ALLOW_ORIGINS` | localhost | Comma-separated |

**No secrets are committed.** `SUPABASE_SECRET_KEY` is server-side only and
bypasses RLS; it must never reach a browser.

---

## Deployment

### Vercel (backend)

`vercel.json` maps every path to the single Python function at `app/main.py`,
where Vercel finds the module-level ASGI `app`. `maxDuration` is 300 s (the Hobby
ceiling; **Pro allows up to 800**, which you want once generation runs on a real
GPU). Notes and rationale: [`VERCEL.md`](VERCEL.md).

After changing runtime dependencies: `make reqs` (regenerates `requirements.txt`,
which is what Vercel installs).

### Supabase

Apply `supabase/migrations/0001_init.sql` via the CLI or the SQL editor. Seven
tables, uuid PKs, jsonb for moving parts, indexes on `generations(created_at desc)`
and `generations(status)`. **RLS is enabled on every table with no policies** —
deny by default; only the service key gets through.

### Modal (GPU)

```bash
modal setup
modal secret create jeno-hf    HF_TOKEN=...
modal secret create jeno-vllm  VLLM_API_KEY=...
modal deploy modal/serve.py          # prints the URL for MODEL_BASE_URL
modal run    modal/train.py --dataset-uri r2://datasets/v1/train.jsonl
```

`serve.py` scales to zero (`min_containers=0`, `scaledown_window=5 min`). If the
adapter directory does not exist yet it serves the base model and logs a warning,
so the endpoint is usable before training lands.

> The directory is named `modal/`, which shadows the installed `modal` package
> when the repo root is on `sys.path`. `modal deploy modal/serve.py` is fine
> (Python puts the script's own directory first); avoid `python -c "import modal"`
> from the repo root.

### Smoke test a deployment

```bash
./scripts/smoke_test.sh https://trend-writer.workser.app
```

---

## Development

```bash
make install   # uv sync --group dev
make dev       # uvicorn with autoreload
make test      # pytest
make lint      # ruff check + ruff format --check + mypy
make fmt       # ruff format + ruff check --fix
make reqs      # regenerate requirements.txt for Vercel
```

CI (`.github/workflows/ci.yml`) runs lint, types, tests, a full mock-provider
smoke test against a live uvicorn, and the bundle-weight guard.
