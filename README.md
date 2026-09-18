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

`make lint test` runs ruff + mypy + the test suite (113 tests).

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
pipeline/            ✅ scrape · clean · label · build_dataset (typer CLIs, resumable)
modal/               ✅ common · train (Unsloth QLoRA) · serve (vLLM + LoRA) · eval (base vs FT)
supabase/migrations/ 0001_init.sql — 7 tables, RLS enabled, no policies
tests/               113 tests: services · API (mock, ASGI) · pipeline · train/inference prompt parity
docs/                architecture.md · report.md (outline) · data_card.md (template)
scripts/             smoke_test.sh · gen_requirements.sh
```

### Done vs stubbed

**Done and tested** — the whole API surface, the A→F generation flow, both
transports, normalization, chunking + BM25 retrieval, the quality gate with its
retry, ingestion of URLs and documents, all three seams with every
implementation, the SQL schema, structured logging, the error model, CI.

**Implemented but not yet run at full scale** — the data pipeline has been run
against the live site on a 12-article sample (discover → crawl → clean →
build → validate all pass); labelling needs your `LABELER_*` endpoint. The Modal
jobs (`train.py`, `serve.py`, `eval.py`) import and construct cleanly against
Modal 1.5.5 but have not executed on a GPU yet. `prompt.STYLE_RULES` is still a
first draft. The browser demo page is not in this repo yet.

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

## Fine-tuning runbook

The whole path from website to adapter. Every stage is resumable: state lives
in `.data/corpus.db` (or `training_articles` with `PERSISTENCE=supabase`), so a
failed run is re-run, not restarted.

```bash
# 0. one-time
uv sync --group dev --group modal
uv run modal setup                          # browser login

# 1. corpus (~3-5 min for 174 articles at 1 req/s)
make corpus                                 # discover -> crawl -> clean -> stats

# 2. reverse-label (any OpenAI-compatible model; set LABELER_* in .env)
uv run python -m pipeline.label run --dry-run   # eyeball ONE label first
make label
uv run python -m pipeline.label audit --sample 10

# 3. dataset
make dataset VERSION=v1                     # build + validate; writes the data card

# 4. train on Modal (L4, QLoRA r=16, 3 epochs)
make train DATASET=r2://datasets/v1/train.jsonl
#   no R2 yet? copy the file into the volume and point at it:
#   uv run modal volume put jeno-models .data/datasets/v1/train.jsonl /datasets/v1/train.jsonl
#   make train DATASET=/models/datasets/v1/train.jsonl

# 5. serve + evaluate
make serve                                  # prints the vLLM URL
make eval ENDPOINT=https://<ws>--jenosize-trend-writer-vllmserver.modal.run/v1
```

Modal secrets the jobs expect: `jeno-hf` (`HF_TOKEN`), `jeno-vllm`
(`VLLM_API_KEY`), `jeno-r2` (`R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`,
`R2_SECRET_ACCESS_KEY`, `R2_BUCKET`).

### What the data pipeline does, and why

| Stage | Key decisions |
|---|---|
| **scrape** | robots.txt honoured, 1 req/s, raw HTML archived *before* parsing. `category` comes free from the URL path (`/en/ideas/{category}/{slug}`), and title + meta description come from the page itself, so they are real training targets. |
| **clean** | Headings remapped relative to each article's shallowest level (the site mixes `<h4>` and `<h5>` layouts). CTA and **reference sections are dropped whole**, because a model that learns to write bibliographies makes up sources at inference. Near-duplicates are flagged by 5-gram shingle Jaccard, not deleted. |
| **label** | An LLM infers topic / industry / audience / keywords from the finished article. Category, language and length are **derived, not inferred**. Every inferred field goes through the API's own normalizers. |
| **build** | Rendered through `app/services/prompt.py`, the same code the API calls. `tests/test_pipeline_dataset.py` asserts the two prompts are **byte-identical**. Split by `sha256(url)` so adding articles never moves one across train/eval. |

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

Apply `supabase/migrations/0001_init.sql` then `0002_training_article_fields.sql`
via the CLI or the SQL editor. Seven
tables, uuid PKs, jsonb for moving parts, indexes on `generations(created_at desc)`
and `generations(status)`. **RLS is enabled on every table with no policies** —
deny by default; only the service key gets through.

### Modal (GPU)

```bash
uv sync --group modal
uv run modal setup
uv run modal secret create jeno-hf    HF_TOKEN=...
uv run modal secret create jeno-vllm  VLLM_API_KEY=...
uv run modal secret create jeno-r2    R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_BUCKET=jenosize-trend-writer
uv run modal deploy modal/serve.py   # prints the URL for MODEL_BASE_URL
```

`serve.py` scales to zero (`min_containers=0`, `scaledown_window=5 min`). If the
adapter directory does not exist yet it serves the base model and logs a warning,
so the endpoint is usable before training lands.

> The `modal/` directory has no `__init__.py`, so it is only a namespace-package
> candidate, and Python prefers the real `modal` SDK from site-packages (checked:
> `import modal` resolves correctly from the repo root). Don't add an
> `__init__.py` there, because that would really shadow the SDK.
>
> Install the Modal CLI into the project venv with `uv sync --group modal`, not
> `pip install modal`. Homebrew's Python refuses system-wide pip installs
> (PEP 668), and the group keeps `modal` out of the Vercel bundle.

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
