# Jenosize Trend Writer — assignment report

**Option 1: Trend & Future Ideas Articles · 23 September 2026**

**Web app:** WEB_URL · **API docs:** https://khemmapich--jenosize-trend-writer-article-api.modal.run/docs · **Code:** REPO_URL

## Approach

The prototype accepts a business topic, category, industry, audience, SEO
keywords, target length, and optional URL or document. A FastAPI service
normalizes the brief, retrieves relevant passages from any supplied source,
prompts a fine-tuned writer, checks the output, and returns an article. A
separate LangChain Studio agent can research the web (parallel Tavily
searches), calls that same writer through a tool, designs a branded page, and
offers a preview and explicit publishing action in a Next.js app. The agent
never substitutes its own prose for the writer's article.

I selected **Qwen3-4B-Instruct-2507** because a 4B instruction model is small
enough to serve on one Modal GPU while retaining useful English business-writing
ability. LoRA rank 16 adapts style without retraining all weights. The deployed
alias `jeno-lora` points to the active `jeno-lora-v2` adapter; the base model is
also served for comparison. Retrieval is kept outside the weights: a small
brand corpus should teach tone, not be trusted as a source of current facts.

## Data engineering and training

The pipeline discovered 174 Jenosize Ideas URLs, fetched 170, cleaned 159,
and reverse-labelled 159 articles into briefs. It strips navigation, CTAs,
bylines, and reference sections; normalizes headings; filters length and
near-duplicates; and validates title, meta description, and section structure.
The v2 quality filter excluded 54 candidates and published **92 train / 13
held-out** examples. Splitting is deterministic by source-URL hash; dataset
fingerprint is `69f0b4147fd8`. The JSONL contains system, user brief, and
assistant article turns, using the same prompt builder as inference. The
dataset and a detailed data card are included in the private submission bundle.

The active v2 adapter trained for **one epoch, 12 optimizer steps**, with LoRA
rank 16 and learning rate `1e-4`; recorded final train loss was **2.2831**.
Loss is not directly comparable to v1 because the corpus, quality filter,
epoch count, and learning rate changed. The training and serving code, not a
hosted third-party fine-tune, loads this adapter in vLLM. The adapter
weights, the per-step loss curve (2.71 → 1.98), the run records and all 26
evaluation outputs are published in the repository's `evidence/` folder.

## Evaluation and observed product flow

On 13 held-out briefs, the live v2 evaluation reported:

| Measure | Base | Fine-tuned v2 |
|---|---:|---:|
| Deterministic quality-gate pass rate | 69.2% (9/13) | **76.9% (10/13)** |
| Mean SEO keyword coverage | **95.8%** | 92.1% |
| Mean article words | 705 | 744 |
| Mean H2 sections | 4.38 | 4.92 |

The independent blind judge recorded **3 fine-tuned wins and 2 base wins**,
but only five parseable comparisons are present, so this is not a 13-pair
style result. The one-brief quality-gate edge is likewise too small to claim
general superiority. v1 performed materially worse than base; v2 is a better
demo checkpoint, not proof that fine-tuning universally improves quality.

Everything is deployed: the article API, jobs/agent API and vLLM run on
Modal, and the Next.js app is on Vercel. In the web app, a user brief causes a
`write_article` agent call, the `jeno-lora` writer produces a stored draft,
and Studio renders and publishes it. The public API exposes
`POST /api/v1/articles` for direct topic-to-article generation and an SSE
streaming variant; a live call returned a 559-word, 3-section article from
`jeno-lora` that passed the quality gate. A warm vLLM reply takes seconds; a scale-to-zero cold start
was observed at roughly 2–3 minutes, so page entry triggers best-effort warmup
and generation retries transient 502/503/504 responses.

## Challenges, limitations, and next steps

The biggest engineering issue was Modal's cold GPU startup returning 503 before
an upstream existed. Bounded readiness polling, retries, and UI progress make
this recoverable, but cannot remove cold-start latency without a paid warm
container. API latency was the second issue: every request opened a fresh
Postgres connection (~0.8 s); a pooled connection plus one warm CPU container
cut list requests from ~2.5 s to ~1 s.

The dataset is small, English-only, single-publisher, and its briefs are
inferred from finished articles. Source-grounded examples were not in the
fine-tuning set. Generated claims still require editorial fact-checking, and
the prototype is browser-scoped rather than production multi-tenant. Next
steps are human review of more held-out topics (especially Thai), a complete
13-pair style adjudication, evidence/citation checks, and a persistent warm
serving tier only if usage justifies its cost. The public demo has no login; it
needs authentication and rate limits before wider use.
