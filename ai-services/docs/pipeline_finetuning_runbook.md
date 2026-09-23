# Jenosize data pipeline: scrape → fine-tune → evaluate

This is the end-to-end runbook for turning Jenosize articles into a private
LoRA adapter and comparing that adapter with the base model.

The important separation is:

- **Postgres/Supabase** stores pipeline state and job status.
- **Cloudflare R2** stores raw HTML, datasets, data cards, and evaluation files.
- **Modal CPU workers** run scraping, cleaning, labeling, and orchestration.
- **Modal GPU workers** run QLoRA training and evaluation.

Fine-tuning teaches the model Jenosize's writing style and structure. It does
not replace retrieval for current facts.

## Pipeline diagram

```mermaid
flowchart LR
    A[CLI bootstrap / local server] --> B[POST scrape]
    B --> C[202 JobRun ID]
    C --> D[Discover sitemaps]
    D --> E[Crawl new or changed pages]
    E --> F[Clean markdown]
    E --> R[(R2 raw HTML)]
    F --> P[(Postgres training_articles)]

    P --> G[POST label preview]
    P --> H[POST label]
    H --> I[LLM reverse-labels articles]
    I --> P

    P --> J[POST datasets]
    J --> K[Validate + split train/eval]
    K --> S[(R2 datasets/v1/*.jsonl)]

    S --> L[POST train on Modal Jobs API]
    L --> M[Modal L4 QLoRA]
    M --> N[(Modal Volume adapter)]
    N --> O[POST adapters/v1/activate]
    O --> V[vLLM base + LoRA]

    S --> Q[POST eval]
    V --> Q
    Q --> T[Deterministic metrics + optional blind judge]
    T --> U[(R2 eval/{run}/results.json)]
```

The same flow in plain text is:

```text
discover → crawl → clean → label → build dataset → train → serve → evaluate
   │          │       │       │          │           │        │        │
 sitemap    R2 raw  markdown labels   train/eval   LoRA     vLLM   metrics
             HTML    in DB             JSONL      volume           + report
```

## 0. Choose the API base URL

There are two supported modes. The endpoint paths differ by one prefix.

| Mode | Base URL | Use |
|---|---|---|
| Local combined API | `http://127.0.0.1:8777/jobs` | Scrape, clean, label, and build a dataset locally. Training/evaluation must still use Modal. |
| Deployed Modal Jobs API | `https://<workspace>--jenosize-trend-writer-jobs-api.modal.run` | Canonical production workflow, including GPU training and evaluation. |

For the local server:

```bash
cd ~/career/jenosize/ai-services
uv run python scripts/dev_server.py --with-jobs
```

It exposes article API routes at `/api/v1/...` and Jobs API routes at
`/jobs/v1/...`. Swagger is available at:

```text
http://127.0.0.1:8777/docs
http://127.0.0.1:8777/jobs/docs
```

For the examples below, set one base URL. Do not put a real API key in this
file or commit one to Git:

```bash
export JOBS_BASE="http://127.0.0.1:8777/jobs"
export JOBS_API_KEY="<your JOBS_API_KEY>"
```

For the deployed workflow, use the URL printed by `make deploy-modal`:

```bash
export JOBS_BASE="https://<workspace>--jenosize-trend-writer-jobs-api.modal.run"
export JOBS_API_KEY="<your JOBS_API_KEY>"
```

Every guarded Jobs API request uses:

```bash
-H "X-API-Key: $JOBS_API_KEY"
```

## 1. One-time CLI bootstrap

These are setup/deployment commands, not data-processing stages:

```bash
uv sync --group dev --group pipeline --group modal
uv run modal setup
make modal-secrets
make deploy-modal
make modal-doctor
```

`make modal-secrets` creates the private Modal secrets. `make deploy-modal`
deploys the Jobs API, CPU workers, vLLM service, train function, and eval
function. Redeployment is only needed after changing Modal-side code or image
dependencies.

## 2. Verify infrastructure and migrations

These calls are safe to repeat:

```bash
curl -sS "$JOBS_BASE/health" | jq

curl -sS "$JOBS_BASE/v1/doctor" \
  -H "X-API-Key: $JOBS_API_KEY" | jq

curl -sS -X POST "$JOBS_BASE/v1/migrations/apply" \
  -H "X-API-Key: $JOBS_API_KEY" | jq
```

`doctor` checks imports, database connectivity, R2 reachability, and labeler
configuration. The migration call creates/records the required tables when
they are missing.

## 3. Scrape, crawl, and clean

### Request

`POST /v1/scrape` returns `202 Accepted` and a `JobRun` ID. It runs three
stages in order: `discover`, `crawl`, and `clean`.

```bash
SCRAPE_RESPONSE=$(curl -sS -X POST "$JOBS_BASE/v1/scrape" \
  -H "X-API-Key: $JOBS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"limit":200,"delay_s":1.0,"recheck_days":null}')

echo "$SCRAPE_RESPONSE" | jq
export SCRAPE_RUN_ID=$(echo "$SCRAPE_RESPONSE" | jq -r '.id')
```

### Parameters

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `limit` | integer, 1–5000 | `1000` | Maximum pages selected for crawling in this run. |
| `delay_s` | number, 0.5–10 | `1.0` | Delay between page requests; protects the source site. |
| `recheck_days` | number or `null` | `null` | Only re-fetch pages older than this many days. `null` means new/never-successfully-checked pages. |

`limit` is not the total number of sitemap URLs. Discovery can find more URLs;
the crawl stage processes only the due rows up to the limit. Known unchanged
articles are skipped.

### Poll the run

```bash
curl -sS "$JOBS_BASE/v1/runs/$SCRAPE_RUN_ID" \
  -H "X-API-Key: $JOBS_API_KEY" | jq
```

Wait for:

```json
"status": "succeeded"
```

The response includes per-stage stats such as `new`, `known`, `checked`,
`failed`, `cleaned`, `too_short`, and `wrong_language`.

### What each stage writes

| Stage | Work | Persistent result |
|---|---|---|
| `discover` | Reads the Jenosize sitemap and keeps English idea URLs. | New URLs in `training_articles`. |
| `crawl` | Fetches due pages with the configured delay. | New/changed raw HTML in R2 and content hashes in Postgres. |
| `clean` | Extracts article markdown, remaps headings, removes CTAs/references, rejects bad language/length/duplicates. | `clean_markdown`, word count, and rejection metadata. |

Inspect the result without changing data:

```bash
curl -sS "$JOBS_BASE/v1/corpus" \
  -H "X-API-Key: $JOBS_API_KEY" | jq

curl -sS "$JOBS_BASE/v1/corpus/stats" \
  -H "X-API-Key: $JOBS_API_KEY" | jq

curl -sS "$JOBS_BASE/v1/corpus/articles?state=rejected&limit=200" \
  -H "X-API-Key: $JOBS_API_KEY" | jq

curl -sS "$JOBS_BASE/v1/corpus/sample?n=10" \
  -H "X-API-Key: $JOBS_API_KEY" | jq
```

Useful article states are `pending`, `fetched`, `cleaned`, `rejected`,
`duplicates`, and `labelled`.

## 4. Reverse-label the cleaned articles

Reverse-labeling asks the labeler LLM to infer the brief that could have
produced each existing article. The labels become the training input, while
the cleaned article remains the assistant output.

The labeler is configured by:

```text
LABELER_BASE_URL
LABELER_API_KEY
LABELER_MODEL
```

### Preview one article (does not write)

```bash
curl -sS -X POST "$JOBS_BASE/v1/label/preview" \
  -H "X-API-Key: $JOBS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{}' | jq
```

The response should contain `url`, `title`, `labels`, `excerpt`, and
`written: false`. This is a quality check, not the full labeling operation.

### Label all pending/changed articles

```bash
LABEL_RESPONSE=$(curl -sS -X POST "$JOBS_BASE/v1/label" \
  -H "X-API-Key: $JOBS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"limit":5000}')

echo "$LABEL_RESPONSE" | jq
export LABEL_RUN_ID=$(echo "$LABEL_RESPONSE" | jq -r '.id')
```

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `limit` | integer, 1–5000 | `1000` | Maximum cleaned articles to label in this run. |
| `model` | string or `null` | configured model | Optional per-run override for `LABELER_MODEL`. |

Poll the run:

```bash
curl -sS "$JOBS_BASE/v1/runs/$LABEL_RUN_ID" \
  -H "X-API-Key: $JOBS_API_KEY" | jq
```

Interpret label stats carefully:

- `labelled` is the number successfully written.
- `failed` is the number of individual articles that failed.
- `pending` is the number selected at the start of that run; it does not mean
  that all those rows are still pending after completion.

Audit saved labels:

```bash
curl -sS "$JOBS_BASE/v1/corpus/sample?n=10&labelled=true" \
  -H "X-API-Key: $JOBS_API_KEY" | jq
```

The operation is incremental: already-labelled articles whose cleaned hash has
not changed are skipped.

## 5. Build and publish the immutable dataset

`POST /v1/datasets` renders each article through the same prompt builder used by
the article API, then validates the JSONL before uploading it to R2.

```bash
curl -sS -X POST "$JOBS_BASE/v1/datasets" \
  -H "X-API-Key: $JOBS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"version":"v1","eval_frac":0.1,"seed":13,"overwrite":false}' | jq
```

### Dataset parameters

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `version` | `vN` string | `v1` | Immutable dataset/adapter namespace. |
| `eval_frac` | number, 0–<0.5 | `0.1` | Fraction held out for evaluation. |
| `seed` | integer | `13` | Stable URL-hash split seed. |
| `overwrite` | boolean | `false` | Only use to replace an untrained existing version. |
| `quality_filter` | boolean | `false` | Curate only training rows that satisfy the production output contract; eval URLs stay fixed. |

Expected success is `201` with `status: "published"`, a train count, an eval
count, and R2 keys. If the corpus has not changed and the same version already
exists, `200`/`status: "unchanged"` is safe. Invalid rows return `422` and
nothing is uploaded.

Validate the exact bytes that Modal will read:

```bash
curl -sS "$JOBS_BASE/v1/datasets/v1/validate" \
  -H "X-API-Key: $JOBS_API_KEY" | jq

curl -sS "$JOBS_BASE/v1/datasets/v1/examples?split=train&n=2" \
  -H "X-API-Key: $JOBS_API_KEY" | jq

curl -sS "$JOBS_BASE/v1/datasets/v1/card" \
  -H "X-API-Key: $JOBS_API_KEY"
```

For the current corpus, the published `v1` dataset contains **146 train** and
**13 eval** examples, with validation passing.

For the next controlled experiment, keep v1 untouched and publish a curated
v2:

```bash
curl -sS -X POST "$JOBS_BASE/v1/datasets" \
  -H "X-API-Key: $JOBS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"version":"v2","eval_frac":0.1,"seed":13,"quality_filter":true}' | jq
```

The response includes `candidates`, `excluded`, and `excluded_reasons`. Review
those counts before starting a GPU run. The filter is intentionally opt-in so
the original v1 benchmark remains reproducible.

## 6. Fine-tune on Modal GPU

Training must use the **deployed Modal Jobs API**, not the local combined API:
the local API has no GPU remote runner. The dataset is read from:

```text
r2://datasets/v1/train.jsonl
```

### Recommended one-shot training request

Use three epochs for the final run. One epoch is only a cheap GPU smoke test.

```bash
TRAIN_RESPONSE=$(curl -sS -X POST \
  "https://<workspace>--jenosize-trend-writer-jobs-api.modal.run/v1/train" \
  -H "X-API-Key: $JOBS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"version":"v1","epochs":3,"lora_r":16,"learning_rate":0.0002}')

echo "$TRAIN_RESPONSE" | jq
export TRAIN_RUN_ID=$(echo "$TRAIN_RESPONSE" | jq -r '.id')
```

### Training parameters

| Parameter | Default | Current recommendation | Purpose |
|---|---:|---:|---|
| `version` | required | `v1` | Reads `datasets/v1/train.jsonl` and writes the v1 adapter. |
| `epochs` | `3` | `3` | Number of passes over 146 training examples. |
| `lora_r` | `16` | `16` | Adapter rank; higher values cost more memory. |
| `learning_rate` | `0.0002` | `0.0002` | LoRA learning rate. |

The underlying Modal training configuration is:

```text
Base model: Qwen/Qwen3-4B-Instruct-2507
GPU: L4
Method: 4-bit QLoRA
Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
Context: 4096 tokens
Per-device batch: 2
Gradient accumulation: 4 (effective batch size 8)
Loss: assistant response only
Seed: 3407
```

Poll training and inspect progress:

```bash
curl -sS \
  "https://<workspace>--jenosize-trend-writer-jobs-api.modal.run/v1/runs/$TRAIN_RUN_ID" \
  -H "X-API-Key: $JOBS_API_KEY" | jq

curl -sS \
  "https://<workspace>--jenosize-trend-writer-jobs-api.modal.run/v1/runs/$TRAIN_RUN_ID/progress" \
  -H "X-API-Key: $JOBS_API_KEY" | jq
```

The adapter is committed to the private Modal volume at a path equivalent to:

```text
/models/jeno-lora-v1
```

If training fails, the scrape, labels, and published dataset remain intact;
only the GPU training job needs to be retried.

## 7. Activate and evaluate

### List adapters

```bash
curl -sS \
  "https://<workspace>--jenosize-trend-writer-jobs-api.modal.run/v1/adapters" \
  -H "X-API-Key: $JOBS_API_KEY" | jq
```

### Activate the adapter alias

```bash
curl -sS -X POST \
  "https://<workspace>--jenosize-trend-writer-jobs-api.modal.run/v1/adapters/v1/activate" \
  -H "X-API-Key: $JOBS_API_KEY" | jq
```

The versioned model name `jeno-lora-v1` remains available immediately. The
short alias `jeno-lora` is updated for the next vLLM cold start.

### Confirm vLLM models

Use the exact vLLM URL printed by `make deploy-modal`:

```bash
export VLLM_BASE="https://<workspace>--jenosize-trend-writer-vllmserver.modal.run/v1"

curl -sS "$VLLM_BASE/models" \
  -H "Authorization: Bearer $VLLM_API_KEY" | jq
```

You should see the base model and the registered adapter name.

### Run base-vs-adapter evaluation

```bash
EVAL_RESPONSE=$(curl -sS -X POST \
  "https://<workspace>--jenosize-trend-writer-jobs-api.modal.run/v1/eval" \
  -H "X-API-Key: $JOBS_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"version\":\"v1\",\"endpoint\":\"$VLLM_BASE\",\"limit\":13,\"judge\":true}")

echo "$EVAL_RESPONSE" | jq
export EVAL_RUN_ID=$(echo "$EVAL_RESPONSE" | jq -r '.id')
```

### Evaluation parameters

| Parameter | Required | Default | Meaning |
|---|---|---:|---|
| `version` | yes | — | Published dataset version whose eval split is replayed. |
| `endpoint` | yes | — | vLLM OpenAI-compatible base URL ending in `/v1`. |
| `judge` | no | `false` | Ask the independent labeler LLM for a blind pairwise comparison. |
| `judge_model` | no | configured labeler | Override the judge model. |
| `judge_endpoint` | no | configured labeler | Override the judge endpoint. |
| `limit` | no | `20` | Number of held-out examples, 1–200. |

Poll the evaluation job:

```bash
curl -sS \
  "https://<workspace>--jenosize-trend-writer-jobs-api.modal.run/v1/runs/$EVAL_RUN_ID" \
  -H "X-API-Key: $JOBS_API_KEY" | jq
```

Evaluation compares the base model and `jeno-lora-v1` using the same briefs,
same vLLM server, and same sampling seed. It reports deterministic quality
checks such as word count, title length, section count, and keyword coverage.
With `judge: true`, an independent labeler model also chooses base, fine-tuned,
or tie and gives a short reason. Per-example outputs are stored under an R2 key
like `eval/{run_id}/results.json`.

## 8. Repeating the workflow later

New articles do not require a full restart:

```text
POST /v1/scrape              # new/changed pages only
POST /v1/label               # new/changed cleaned rows only
POST /v1/datasets {version:v2}
POST /v1/train    {version:v2}
POST /v1/eval     {version:v2, endpoint:...}
POST /v1/adapters/v2/activate
```

Dataset versions are immutable. Use `v2` when the corpus changes or when a new
adapter experiment must not overwrite the v1 adapter.

## Current completed checkpoint

At the time this runbook was written, the pipeline had completed:

```text
174 discovered
170 fetched
159 cleaned
159 labelled
146 train examples
13 eval examples
dataset v1 validation: ok=true, problems=[]
```
