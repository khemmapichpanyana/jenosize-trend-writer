# Fine-tuning workflow

From jenosize.com/en/ideas to a served, evaluated LoRA adapter. It's written for
an engineer, not a data scientist: every step is a command, every command can be
re-run safely, and nothing lives only on a laptop.

```
                 ┌──────────────── your machine (CPU, cheap) ────────────────┐
jenosize.com ──► │ scrape ──► clean ──► label ──► build_dataset              │
  (sitemaps)     └────┬──────────┬────────┬────────────┬─────────────────────┘
                      │          │        │            │
               raw HTML│   rows +  │  rows + │   train/eval│ JSONL + data card
                      ▼   markdown▼  briefs ▼            ▼
            ┌─────────────────┐  ┌──────────────────────────────┐
            │ Cloudflare R2   │  │ Postgres (Supabase)          │
            │ raw/scrape/...  │  │ training_articles            │
            │ datasets/v1/... │  │ pipeline_runs  dataset_versions
            └────────┬────────┘  └──────────────────────────────┘
                     │ r2://datasets/v1/train.jsonl
                     ▼
            ┌──────────────── Modal (GPU, pay per second) ───────────────┐
            │ train.py (L4, QLoRA) ──► Volume jeno-models/jeno-lora-v1    │
            │ serve.py (vLLM: base + LoRA) ◄── eval.py (base vs tuned)    │
            └────────────────────────────────────────────────────────────┘
```

**Rule of thumb:** Postgres holds *state* (what we have, what each stage has
done), R2 holds *bytes* (raw pages, published datasets), Modal holds *GPU work*.

---

## 0. One-time setup (~20 min)

### Accounts and secrets

| Service | What to create | Goes into |
|---|---|---|
| Supabase | A project (region: Singapore). Dashboard → **Connect** → *Session pooler* URI | `.env` → `DATABASE_URL` |
| Cloudflare R2 | Bucket `jenosize-trend-writer` (public access off) + an API token scoped to it, *Object Read & Write* | `.env` → `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` |
| Any OpenAI-compatible LLM | For reverse-labelling (~150 short calls) | `.env` → `LABELER_BASE_URL`, `LABELER_API_KEY`, `LABELER_MODEL` |
| Hugging Face | A token with *write* access (to publish the adapter) | Modal secret `jeno-hf` |
| Modal | `uv run modal setup` (browser login) | `~/.modal.toml` |

Use Supabase's **Session pooler** URI rather than the direct connection. On the
free plan the direct host is IPv6-only, and many home and office networks
aren't. The code disables prepared statements, so the Transaction pooler
(port 6543) also works.

### Install and migrate

```bash
uv sync --group dev --group pipeline --group modal    # or: make install
cp .env.example .env                                  # fill in the values above
make migrate                                          # creates all tables (idempotent)
```

### Modal secrets (the GPU jobs read these, not your `.env`)

```bash
uv run modal secret create jeno-hf   HF_TOKEN=hf_...
uv run modal secret create jeno-vllm VLLM_API_KEY=$(openssl rand -hex 24)
uv run modal secret create jeno-r2   R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... \
                                     R2_SECRET_ACCESS_KEY=... R2_BUCKET=jenosize-trend-writer
```

Don't `pip install modal`. Homebrew's Python refuses system-wide installs, and
the `modal` uv group keeps the CLI out of the Vercel bundle. Prefix every Modal
command with `uv run`.

---

## 1. Gather the articles: `make scrape` (~4 min the first time)

```bash
make scrape      # = pipeline.scrape discover + pipeline.scrape crawl
make status      # progress per stage + the last few runs
```

| Step | What it does | What it writes |
|---|---|---|
| `discover` | Reads both sitemaps, keeps `/en/ideas/{category}/{slug}` | **New URLs only** into `training_articles` (category taken from the URL) |
| `crawl` | Fetches pages not yet checked, 1 request/s, robots.txt honoured | Raw HTML to R2 **only if the content is new or changed**, plus title and meta description to Postgres |

### How "only save new data" works

Measured on the live site before designing it:

| Candidate change signal | Usable? | Why |
|---|---|---|
| Sitemap `<lastmod>` | ✗ | Identical for all 174 articles: it's the site's build time |
| `ETag` / `Last-Modified` headers | ✗ | Not sent |
| Raw HTML hash | ✗ | Changes on *every* request: Cloudflare rewrites its email-protection tokens |
| **Extracted article text hash** | ✓ | Byte-identical across fetches of an unchanged page |

So each page gets a **fingerprint**: the sha256 of its extracted text. On
re-fetch:

- **Same fingerprint:** only `last_checked_at` is updated. No R2 upload, and
  nothing downstream re-runs.
- **New fingerprint:** the raw HTML is archived at
  `raw/scrape/{date}/{fingerprint}.html`, and the old version is kept.
- **Truncated response** (HTTP 200 but no `</html>`): retried and never
  stored. This was observed live, where a cut-off page would otherwise have
  replaced a complete article.

A normal `make scrape` doesn't re-fetch known pages at all, so a run with
nothing new makes zero page requests. To catch edits to existing articles:

```bash
make recheck DAYS=30    # re-fetch pages last checked more than 30 days ago
```

### Failures

| Failure | Recorded as | Retried? |
|---|---|---|
| Timeout / 5xx / truncated | `error`, not marked checked | Yes, automatically on the next run |
| 4xx / page with no article text | `error`, marked checked | Only on `make recheck` |

jenosize.com does return sporadic 500s. The URLs came back 200 minutes later,
so just re-run.

---

## 2. Clean: `make clean` (~seconds)

Reads raw HTML **from R2** and writes `clean_markdown` to Postgres. It only
processes articles whose fingerprint changed, or all of them after a rules
change.

What it removes, and why it matters for the fine-tune:

| Rule | Prevents the model from learning to… |
|---|---|
| Headings remapped so each article's top level becomes `##` | …emit `####` (the site mixes `<h4>` and `<h5>` layouts) |
| "Call to Action" sections and CTA paragraphs dropped | …advertise "contact Jenosize" inside a client article |
| **References / Sources sections dropped** | …write bibliographies, which at inference means *inventing* sources |
| Restated title removed from the body | …print the title twice |
| < 300 or > 4000 words rejected | …stop early, or exceed the context |
| Near-duplicates flagged (5-gram Jaccard ≥ 0.6) | …overweight repeated content |

**Changed a cleaning rule?** Bump `CLEAN_VERSION` in `pipeline/clean.py`. The
next `make clean` re-cleans every article from the archived HTML, with no
re-crawl, and **doesn't** re-bill labelling, because labels are keyed to
content, not to cleaning rules.

```bash
uv run python -m pipeline.clean stats     # counts, word distribution, categories
```

---

## 3. Reverse-label: `make label` (~2–5 min, a few cents)

The corpus has articles but no briefs, and training needs *(brief → article)*
pairs. An LLM reads each finished article and infers the brief that would have
produced it.

```bash
uv run python -m pipeline.label run --dry-run      # ALWAYS eyeball one first
make label
uv run python -m pipeline.label audit --sample 10  # read 10 at random
```

| Field | Source |
|---|---|
| `category` | URL path (the publisher's own taxonomy) |
| `language` | URL path |
| `length` | Actual word count |
| `topic`, `industry`, `audience`, `keywords` | LLM, then passed through **the API's own normalizers** |

It only labels articles whose cleaned content changed since they were last
labelled. **Changed the labelling prompt?** Bump `LABEL_VERSION` in
`pipeline/label.py`.

---

## 4. Publish a dataset: `make dataset VERSION=v1` (~seconds)

Renders every usable article through `app/services/prompt.py`, **the same
code the API uses at inference**. `tests/test_pipeline_dataset.py` fails if the
training and inference prompts ever differ by a single byte. If they drifted,
the adapter would be optimised for a prompt production never sends.

Then it **validates before uploading** (roles, `TITLE:`/`META:` parseable by
production code, `##` sections present, length limits, and no train/eval
overlap) and publishes to R2:

```
datasets/v1/train.jsonl   datasets/v1/eval.jsonl   datasets/v1/data_card.md
```

**Versions are immutable.** A model trained on `v1` must always mean the same
bytes:

| Situation | Result |
|---|---|
| Corpus unchanged since `v1` | `status=unchanged`, nothing written |
| Corpus changed, same version | **Refused**. Publish `v2`, or `--overwrite` if nothing was trained on `v1` yet |
| New version | Published, and recorded in `dataset_versions` with its fingerprint |

The train/eval split is by `sha256(url)`, so adding articles later never moves
an existing one between splits. Moving one would leak eval data into training.

---

## 5. Train on Modal: `make train VERSION=v1` (~15–30 min on an L4)

```bash
make train VERSION=v1
```

Your machine only submits the job. Modal starts an L4 GPU, runs
`modal/train.py`, and shuts the GPU down when it finishes, billing per second.

| Setting | Value | Why |
|---|---|---|
| Base | `Qwen/Qwen3-4B-Instruct-2507` | Small, strong, non-thinking (no `<think>` blocks to leak into training) |
| Method | QLoRA: 4-bit base, LoRA r=16, α=16, all attention + MLP projections | Style transfer fits on one 24 GB L4 |
| Schedule | lr 2e-4, 3 epochs, effective batch 8, max 4096 tokens | Standard QLoRA recipe; change it only on eval evidence |
| Loss | **Assistant turn only** (masked via Qwen's ChatML markers, verified against the model's template) | Otherwise it learns to write *briefs* |

The adapter is saved to the Modal volume `jeno-models` at `/models/jeno-lora-v1`.
To also publish it to Hugging Face:

```bash
uv run modal run modal/train.py --dataset-uri r2://datasets/v1/train.jsonl --push-to-hub
```

The first run is slower (~10 min extra) because it builds the image and
downloads the 8 GB base model into a cache volume. Later runs reuse both.

---

## 6. Serve and evaluate

```bash
make serve                                  # prints https://<ws>--jenosize-trend-writer-vllmserver.modal.run
make eval ENDPOINT=https://<that-url>/v1 VERSION=v1
```

`serve.py` runs one vLLM server holding **both** the base model and the LoRA,
so the eval compares them on identical hardware and sampling. The adapter is
the only variable.

`eval.py` replays the **held-out** briefs from `datasets/v1/eval.jsonl` against
both models and reports:

- **Deterministic scores**, using the live API's own quality gate: pass rate,
  keyword coverage, word count, `##` count and title length.
- **Optional pairwise LLM judge** (`--judge-model …`): blind, with the order
  randomised to cancel position bias.

Put the resulting table in `docs/report.md`. It's the evidence for the
fine-tuning grade.

Then point the API at the endpoint: `MODEL_PROVIDER=openai_compatible`,
`MODEL_BASE_URL=<url>/v1`, `MODEL_NAME=jeno-lora`.

---

## Re-running later (new articles on the site)

```bash
make pipeline               # scrape + clean + label, only what's new
make dataset VERSION=v2     # "unchanged" if nothing new; otherwise a new immutable version
make train VERSION=v2
make eval ENDPOINT=... VERSION=v2
```

Every stage is incremental and every run is logged in `pipeline_runs`, so
`make status` answers "what did the last run actually do?" without digging
through logs.

---

## Where things live

| Thing | Location |
|---|---|
| Article state (URL, fingerprint, markdown, labels, split) | Postgres `training_articles` |
| Run history | Postgres `pipeline_runs` |
| Published dataset versions | Postgres `dataset_versions` + R2 `datasets/{version}/` |
| Raw HTML (every distinct version) | R2 `raw/scrape/{date}/{fingerprint}.html` |
| Trained adapter | Modal volume `jeno-models` → `/models/jeno-lora-v1` (+ HF Hub) |
| Base-model cache | Modal volume `jeno-hf-cache` |

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `The pipeline writes to Postgres + R2. Missing in .env: …` | Fill in those variables. The pipeline deliberately has no local fallback |
| `connection … failed: Network is unreachable` | You're using Supabase's direct (IPv6) host; switch to the Session pooler URI |
| `crawl: … failed=N` | Usually transient 500s from the site. Run `make scrape` again |
| `build` refuses with "immutable" | The corpus changed since that version was published. Use the next version number |
| `validate` reports "no '##' sections" | A new page layout. Inspect it, adjust `pipeline/clean.py`, bump `CLEAN_VERSION` |
| Training fails while building the image | Unsloth and TRL move fast. Pin `unsloth==<last good>` in `modal/common.py` |
| `modal: command not found` | Use `uv run modal …` |
