# Jenosize Trend Writer

Generates business trend and future-ideas articles in the tone of
[Jenosize Ideas](https://www.jenosize.com/en/ideas). You give it a topic, category, industry,
audience, SEO keywords, and optionally a source URL or document. It returns
a structured article written by **Qwen3-4B-Instruct-2507 fine-tuned with LoRA**
on Jenosize's own articles.

> **Design principle: fine-tuning teaches style, retrieval supplies facts.**

## Try it

| What | Link |
| --- | --- |
| Web app (chat with the content agent, preview, publish) · login shared privately | https://jenosize-trend-writer.vercel.app |
| Article API: FastAPI, interactive docs | https://khemmapich--jenosize-trend-writer-article-api.modal.run/docs |
| In-app docs (how it works, fine-tuning, results) | https://jenosize-trend-writer.vercel.app/docs |

The GPU scales to zero when idle, so **the first request after a quiet spell
takes 1–3 minutes** while the model loads. The web app wakes it on page load,
and a status indicator shows whether the model is ready or still starting.

### Call the API

```bash
API=https://khemmapich--jenosize-trend-writer-article-api.modal.run

# Optional: wake the GPU first (returns once the model answers)
curl -s -X POST $API/api/v1/model/warmup

curl -s -X POST $API/api/v1/articles -H 'Content-Type: application/json' -d '{
  "topic": "Agentic AI in Thai retail",
  "category": "Futurist",
  "industry": "ecommerce",
  "audience": "C-suite executives",
  "keywords": ["agentic ai", "retail media"],
  "length": "short"
}' | jq '{status, model_version, title, quality_report}'

# Same request, streamed token by token (SSE)
curl -N -s -X POST $API/api/v1/articles/stream -H 'Content-Type: application/json' \
  -d '{"topic":"The future of embedded finance","industry":"fintech","length":"short"}'
```

Only `topic` is required. `source_url` (or an uploaded document) is chunked
and BM25-retrieved into the prompt, so facts come from the source rather than
from the model's memory.

## Repository layout

| Folder | What's inside |
| --- | --- |
| [`ai-services/`](ai-services/README.md) | Python: FastAPI article API (`app/`), data pipeline (`pipeline/`), fine-tuning, serving and evaluation on Modal (`modal/`), the LangChain content agent (`studio/`), SQL schema, tests |
| [`web/`](web/README.md) | Next.js 16 app: agent chat, article preview and publishing, pipeline/training/model dashboards, in-app docs |
| [`evidence/`](evidence/README.md) | Fine-tuning evidence: the adapter weights, training loss curve and run records, base-vs-fine-tuned evaluation outputs |

## How it works

```
brief ─► normalize + validate ─► (optional) fetch source → chunk → BM25 top-k
      ─► prompt builder (the same code that rendered the training data)
      ─► vLLM on Modal: Qwen3-4B + LoRA adapter "jeno-lora"
      ─► quality gate (title, meta, H2 structure, keyword coverage) → one retry
      ─► article JSON / SSE stream, saved to Postgres + R2
```

- **Data pipeline** (`ai-services/pipeline/`): sitemap discovery → polite
  crawl (raw HTML archived to R2) → clean (strip nav, CTAs, bylines, reference
  lists) → reverse-label each article into the brief that would have produced
  it → versioned, validated JSONL dataset. Every stage is incremental: it only
  processes rows that changed since the last run.
- **Fine-tuning** (`ai-services/modal/train.py`): Unsloth QLoRA on a Modal L4,
  rank 16. Training prompts are rendered by the same `app/services/prompt.py`
  used at inference, and a test enforces that they are byte-identical.
- **Serving** (`ai-services/modal/serve.py`): vLLM's OpenAI-compatible server
  with base model + every adapter version by name, plus the `jeno-lora` alias
  for the active one. It scales to zero.
- **Agent** (`ai-services/studio/`): a LangChain agent that can research the
  web (Tavily, parallel searches), call the fine-tuned writer as a tool, add a
  GPT Image 2.5 hero image (generated while the article is written), design a
  branded HTML page, and publish it. Every step (inputs, progress, results,
  timing) is shown live in the chat and saved with the conversation. The agent never substitutes its own
  prose for the writer's article.

Details: [`ai-services/docs/architecture.md`](ai-services/docs/architecture.md),
[`ai-services/docs/fine_tuning_workflow.md`](ai-services/docs/fine_tuning_workflow.md),
[`ai-services/docs/report.md`](ai-services/docs/report.md).

## Dataset and results

- 174 URLs discovered on Jenosize Ideas → 170 fetched → 159 cleaned and labelled.
- v2 (active) dataset: quality-filtered to **92 train / 13 held-out**, split
  deterministically by URL hash (fingerprint `69f0b4147fd8`). The data card is
  in [`ai-services/docs/data_card.md`](ai-services/docs/data_card.md).
- The dataset files are **not in this public repo** because they contain full
  text of a third party's published articles. They are shared privately with the
  submission, and the pipeline above rebuilds them from the public site.
- The v2 adapter was trained with QLoRA for 1 epoch (12 steps) at rank 16 and
  learning rate 1e-4. Loss went from 2.71 to 1.98. **The weights, loss curve,
  run records and every evaluation output are in [`evidence/`](evidence/README.md).**

| On 13 held-out briefs | Base | Fine-tuned v2 |
| --- | --- | --- |
| Quality-gate pass rate | 69.2% (9/13) | 76.9% (10/13) |
| SEO keyword coverage | 95.8% | 92.1% |
| Mean words | 705 | 744 |
| Mean H2 sections | 4.38 | 4.92 |

An earlier v1 adapter (3 epochs, unfiltered data) scored only 15% and lost
every judged comparison. v2's data filter and gentler schedule fixed that.
This is a small, honest comparison. It shows v2 follows the house structure
more reliably, but it does not prove general superiority. See the report for
the judge results and limitations.

## Run it locally

Backend (no accounts, GPU or keys needed; it uses the mock model):

```bash
cd ai-services
uv sync --group dev
cp .env.example .env
make dev                  # http://localhost:8777/docs
./scripts/smoke_test.sh   # health + generate + stream
make lint test            # ruff + mypy + pytest
```

Web app (points at a Studio API, local or deployed):

```bash
cd web
cp .env.example .env.local   # set STUDIO_API_URL and STUDIO_API_KEY
npm install
npm run dev                  # http://localhost:3030
```

Deploying to your own Modal, Supabase and R2 accounts:
[`ai-services/README.md`](ai-services/README.md) and
[`web/DEPLOYMENT.md`](web/DEPLOYMENT.md).

## Scaling and next steps

- The GPU scales to zero to save cost. Setting `min_containers=1` on the vLLM
  server removes cold starts when traffic justifies it. vLLM batches
  concurrent requests and serves every adapter from one base model.
- The API layer is stateless and CPU-only, so it scales horizontally. Long work
  (crawls, training, agent turns) runs as background jobs with resumable
  event streams.
- Next: more and multilingual (Thai) training data, a larger human-judged
  evaluation, citation and fact checks on generated claims, and authentication
  plus rate limits before opening the app beyond a demo.
