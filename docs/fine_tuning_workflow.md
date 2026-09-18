# Fine-tuning workflow (API-driven)

From jenosize.com/en/ideas to a served, evaluated and published LoRA adapter.
**After a one-time bootstrap of three commands, every step is an HTTP call** to
the jobs API. Every call can be repeated safely, and nothing lives only on a
laptop.

- Ready-to-send requests for every step: [`docs/jobs_api.http`](jobs_api.http)
  (VS Code REST Client or JetBrains).
- Interactive docs: `<jobs-url>/docs` (Swagger).

```
                  POST /v1/scrape · /v1/label · /v1/datasets
                 ┌──────────── Modal job worker (CPU, cheap) ────────────────┐
jenosize.com ──► │ scrape ──► clean ──► label ──► build dataset              │
  (sitemaps)     └────┬──────────┬────────┬────────────┬─────────────────────┘
                raw HTML│  rows +  │ rows +  │   train/eval│ JSONL + data card
                      ▼  markdown▼ briefs  ▼            ▼
            ┌─────────────────┐  ┌──────────────────────────────┐
            │ Cloudflare R2   │  │ Postgres (Supabase)          │
            │ raw/scrape/...  │  │ training_articles  job_runs  │
            │ datasets/v1/... │  │ pipeline_runs dataset_versions
            └────────┬────────┘  └──────────────────────────────┘
                     │ r2://datasets/v1/train.jsonl   POST /v1/train · /v1/eval · /adapters/…
                     ▼
            ┌──────────────── Modal (GPU, pay per second) ─────────────────┐
            │ train (L4, QLoRA) ──► volume jeno-models/jeno-lora-{version}  │
            │ vLLM: base + jeno-lora-v1, -v2 … + `jeno-lora` (active)       │
            └──────────────────────────────────────────────────────────────┘
```

Postgres holds *state* (what we have and what each stage did), R2 holds
*bytes* (raw pages and datasets), and Modal holds *compute*. The article API on
Vercel is a separate service: it never scrapes or trains, and only shares the
database and the bucket.

---

## 0. Bootstrap (once, ~15 min): the only CLI steps

The API can't deploy itself, so these three steps run from a terminal:

```bash
uv sync --group dev --group pipeline --group modal
uv run modal setup          # 1. log in to Modal (browser)
make modal-secrets          # 2. create the Modal secrets from .env
make deploy-modal           # 3. deploy jobs API + vLLM + train/eval/publish
```

`.env` needs these keys before step 2. `make modal-secrets` gives each Modal
secret only the keys it uses, and never prints a value.

| Keys | Used by |
|---|---|
| `SUPABASE_URL`, `DB_PASSWORD`, `DB_HOST` (session pooler) | jobs API + workers (Postgres). `DB_URL` is used locally; Modal gets the IPv4 pooler instead |
| `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_API_ENDPOINT`, `R2_JENOSIZE_BUCKET` | everything that reads or writes R2 |
| `JOBS_API_KEY` (`openssl rand -hex 24`) | the `X-API-Key` header on every `/v1` call |
| `LABELER_BASE_URL`, `LABELER_API_KEY`, `LABELER_MODEL` | reverse-labelling (any OpenAI-compatible model) |
| `HF_TOKEN` (write), `VLLM_API_KEY` (`openssl rand -hex 24`) | publishing, and the vLLM server's bearer token |

Before step 3, turn off public access on the bucket (Cloudflare → R2 → bucket
→ Settings → Public Development URL → Disable). It holds scraped third-party
articles and user uploads. The API hands out temporary signed links when a
person needs to open a raw page.

Everything after this point is HTTP. In the examples below, `$JOBS` is the URL
printed by `make deploy-modal`, and every `/v1` call sends
`-H "X-API-Key: $KEY"`.

### Verify and migrate

```bash
curl $JOBS/v1/doctor -H "X-API-Key: $KEY"                  # database, R2, labeller, adapters: all ✓?
curl -X POST $JOBS/v1/migrations/apply -H "X-API-Key: $KEY" # idempotent; records what is already there
```

`GET /v1/config` shows what the server resolved from its secrets: hosts and
names, never values.

---

## Jobs: how every long-running call behaves

`POST /v1/scrape`, `/v1/label`, `/v1/train`, `/v1/eval` and
`/v1/adapters/{v}/publish` return **`202` with a run id straight away**. The
work runs in its own Modal container. Poll `GET /v1/runs/{id}` to follow it:
the response shows the status, the result, and stats for each stage.

| Behaviour | Why |
|---|---|
| **One active job per kind** (`409` otherwise, with the running id), enforced by a unique index | Two crawls would double-fetch every page; two trainings would race for the same adapter directory |
| A worker that dies without reporting is marked `failed` on the next read | A crashed job must not block new ones forever |
| Every request is refused if `JOBS_API_KEY` is unset | These endpoints spend GPU time and labelling budget |
| `POST /v1/runs/{id}/cancel` | Stops the worker, and the GPU call it is waiting on |

---

## 1. Pull the articles: `POST /v1/scrape` (~4-6 min the first time)

One job runs **discover → crawl → clean**:

| Stage | What it does | What it writes |
|---|---|---|
| discover | Reads both sitemaps and keeps `/en/ideas/{category}/{slug}` | **New URLs only** (the category comes from the URL) |
| crawl | Fetches pages not yet checked, 1 req/s, robots.txt honoured | Raw HTML to R2 **only if the content is new or changed**, plus the page's own title and meta description |
| clean | Remaps headings; drops CTA and reference sections, duplicate titles and Thai pages | `clean_markdown`, `word_count`, near-duplicate flags |

To catch edits to existing articles, send `{"recheck_days": 30}`, or `0` to
re-check everything. The crawl delay can't go below 0.5 s, so the API can't be
used to hammer the site.

### How "only save new data" works

This was measured on the live site before it was designed:

| Candidate change signal | Usable? | Why |
|---|---|---|
| Sitemap `<lastmod>` | ✗ | Identical for all 174 articles: it's the site's build time |
| `ETag` / `Last-Modified` | ✗ | Not sent |
| Raw HTML hash | ✗ | Changes on every request (Cloudflare email-protection tokens) |
| **Extracted article text hash** | ✓ | Byte-identical across fetches of an unchanged page |

- **Same fingerprint:** only `last_checked_at` is updated. Nothing downstream
  re-runs.
- **New fingerprint:** the raw HTML is archived at
  `raw/scrape/{date}/{fingerprint}.html`, and the old version is kept.
- **Truncated response** (HTTP 200 but no `</html>`): retried and never stored.
  This was observed live, where a cut-off page replaced a complete article.
- **Failures:** timeouts, 5xx and truncated pages are retried on the next
  scrape. A 4xx or a page with no article text is retried only on a re-check.

## 2. Inspect the corpus (all read-only)

| Question | Call |
|---|---|
| How far along is each stage? | `GET /v1/corpus` |
| Word counts, categories, why articles were rejected | `GET /v1/corpus/stats` |
| List articles by state | `GET /v1/corpus/articles?state=rejected` (`pending`, `fetched`, `cleaned`, `rejected`, `duplicates`, `labelled`) |
| Read one article in full, with a signed link to its raw HTML | `GET /v1/corpus/article?url=…` |
| A random sample to review cleaning quality | `GET /v1/corpus/sample?n=3` |

**What the cleaning removes, and why it matters for the fine-tune:**

| Rule | Prevents the model from learning to… |
|---|---|
| Headings remapped so each article's top level becomes `##` | …emit `####` (the site mixes `<h4>` and `<h5>` layouts) |
| "Call to Action" sections and CTA paragraphs dropped | …advertise "contact Jenosize" inside a client article |
| **References / Sources sections dropped** | …write bibliographies, which at inference means *inventing* sources |
| Restated title removed from the body | …print the title twice |
| **Thai-language pages rejected by language** | Found live: some `/en/` URLs serve Thai. Word counts are meaningless for Thai, so the old "too short" rejection had the right result for the wrong reason |
| < 300 or > 4000 words; near-duplicates (5-gram Jaccard ≥ 0.6) | …stop early, exceed the context, or overweight repeated content |

Changing a rule means bumping `CLEAN_VERSION` in `pipeline/clean.py` and
redeploying. The next scrape job re-cleans every article from the archived
HTML: no re-crawl, and no re-labelling.

## 3. Reverse-label

```bash
curl -X POST $JOBS/v1/label/preview -H "X-API-Key: $KEY"           # ONE article, not saved: check it
curl -X POST $JOBS/v1/label         -H "X-API-Key: $KEY"           # job: every new/changed article
curl "$JOBS/v1/corpus/sample?n=10&labelled=true" -H "X-API-Key: $KEY"   # audit 10 at random
```

An LLM reads each finished article and infers the brief that would have
produced it:

| Field | Source |
|---|---|
| `category`, `length` | URL path and actual word count (derived, not inferred) |
| `language` | The article's own script (not the URL) |
| `topic`, `industry`, `audience`, `keywords` | LLM, then passed through the API's own normalizers |

Only articles whose cleaned content changed since they were last labelled get
labelled again. Changing the prompt means bumping `LABEL_VERSION`.

## 4. Publish a dataset: `POST /v1/datasets {"version": "v1"}` (seconds, synchronous)

The dataset is rendered through `app/services/prompt.py`, **the same code the
article API uses at inference**. A test fails if the training and inference
prompts ever differ by one byte. It's validated *before* upload, then published
to `datasets/v1/` in R2.

| Situation | Response |
|---|---|
| New version | `201` published, recorded with its fingerprint |
| Corpus unchanged since `v1` | `200` unchanged, nothing written |
| Corpus changed, same version | `409`: versions are **immutable**; publish `v2` (or `"overwrite": true` if nothing was trained on it) |
| Invalid rows | `422` with the problems, nothing uploaded |

Then check it: `GET /v1/datasets/v1/validate`, `GET /v1/datasets/v1/card`
(the data card) and `GET /v1/datasets/v1/examples` (rows exactly as the trainer
sees them). The split is by `sha256(url)`, so new articles never move an
existing one between train and eval.

## 5. Train: `POST /v1/train {"version": "v1"}` (~15-30 min on an L4)

Start with `{"version": "v1", "epochs": 1}`, which costs about $0.50 and proves
the GPU path before the full run.

| Setting | Default | Why |
|---|---|---|
| Base | `Qwen/Qwen3-4B-Instruct-2507` | Small, strong, non-thinking: no `<think>` blocks to leak into training |
| Method | QLoRA: 4-bit base, all attention + MLP projections | Style transfer fits on one 24 GB L4 |
| `lora_r` / `epochs` / `learning_rate` | 16 / 3 / 2e-4 | Standard QLoRA recipe; change it only with eval evidence. `lora_r` up to 64 |
| Loss | Assistant turn only (Qwen ChatML markers, verified against the model's template) | Otherwise it learns to write *briefs* |

The adapter lands at `/models/jeno-lora-{version}` on the Modal volume, **one
directory per version**, so v2 never overwrites v1. `GET /v1/adapters` lists
them with their training loss.

## 6. Serve and evaluate

The vLLM server registers **every** trained version under its own name
(`jeno-lora-v1`, `jeno-lora-v2`, …), plus `jeno-lora` for the active one. It
allows ranks up to 64, where vLLM's default of 16 would refuse a rank-32
adapter.

```bash
curl -X POST $JOBS/v1/adapters/v1/activate -H "X-API-Key: $KEY"   # jeno-lora -> v1 (next cold start)
curl -X POST $JOBS/v1/eval -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
     -d '{"version":"v1","endpoint":"https://<ws>--jenosize-trend-writer-vllmserver.modal.run/v1"}'
```

Eval replays v1's **held-out** briefs against the base model and
`jeno-lora-v1` on the same server. The adapter is the only thing that
differs. It reports:

- **The live API's own quality gate:** pass rate, keyword coverage, word
  count, `##` count and title length.
- **Optional blind pairwise judge** (`"judge": true`): the labelling LLM, not
  the model under test, compares the two outputs; presentation order is
  randomised to cancel position bias.

Per-example outputs go to `eval/{run}/results.json` in R2. Put the summary table
in `docs/report.md`.

The article API then points at it with `MODEL_PROVIDER=openai_compatible`,
`MODEL_BASE_URL=<vllm>/v1` and `MODEL_NAME=jeno-lora`.

## 7. Publish: `POST /v1/adapters/v1/publish {"repo_id": "you/jeno-trend-writer-lora"}`

This is a job that uploads the evaluated adapter to Hugging Face. It's separate
from training, so only an adapter that has been evaluated gets published.

---

## Re-running later (new articles on the site)

```
POST /v1/scrape                            only new/changed articles are written
POST /v1/label                             only new/changed articles are labelled
POST /v1/datasets {"version":"v2"}         200 "unchanged" if nothing new, else 201
POST /v1/train    {"version":"v2"}         -> /models/jeno-lora-v2 (v1 untouched)
POST /v1/eval     {"version":"v2", …}      base vs jeno-lora-v2
POST /v1/adapters/v2/activate              if it wins
```

## Local development (optional)

`make jobs-dev` runs the same API on `http://localhost:8001` against your real
Postgres and R2, with scrape and label running in-process (train, eval and
publish need Modal). The `make scrape|clean|label|dataset` targets still exist
for debugging a single stage; they call the same code as the API.

## Where things live

| Thing | Location |
|---|---|
| Article state (URL, fingerprint, markdown, labels, split) | Postgres `training_articles` |
| Jobs / stage runs | Postgres `job_runs` / `pipeline_runs` |
| Published datasets | Postgres `dataset_versions` + R2 `datasets/{version}/` |
| Raw HTML (every distinct version) | R2 `raw/scrape/{date}/{fingerprint}.html` |
| Trained adapters | Modal volume `jeno-models` → `/models/jeno-lora-{version}`, plus `ACTIVE_ADAPTER` |
| Eval outputs | R2 `eval/{run}/results.json` |

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `503 not_configured` | A secret is missing: `GET /v1/doctor` names it; fix `.env`, `make modal-secrets`, redeploy |
| Doctor: `pending migrations` | `POST /v1/migrations/apply` |
| `409 job_already_active` | One job per kind: poll the `run_id` in the response, or cancel it |
| Many `rejected` articles | `GET /v1/corpus/articles?state=rejected` shows each reason; `GET /v1/corpus/stats` groups them |
| `POST /v1/datasets` → `409` | The corpus changed since that version: publish the next version number |
| `POST /v1/eval` → `404` | No trained adapter for that version yet: train first |
| Training fails while building the image | Unsloth and TRL move fast: pin `unsloth==<last good>` in `modal/common.py` |
| Wrong bucket or credentials locally | `.env` beats shell exports, so check `GET /v1/config` shows the expected bucket |
