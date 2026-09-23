# Jenosize Trend Writer — assignment report

**Option 1: Trend & Future Ideas Articles · 22 September 2026**

## Approach

The prototype accepts a business topic, category, industry, audience, SEO
keywords, target length, and optional URL or document. A FastAPI service
normalizes the brief, retrieves relevant passages from any supplied source,
prompts a fine-tuned writer, checks the output, and returns an article. A
separate LangChain Studio agent calls that same writer through a tool, then
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
hosted third-party fine-tune, loads this adapter in vLLM.

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

The tested end-to-end path used a real local FastAPI/Next.js pair and the
remote Modal vLLM: a user brief caused a `write_article` agent call, the
`jeno-lora` writer produced a stored draft, and Studio rendered its preview.
The draft appears in the searchable article library. FastAPI also exposes
`POST /api/v1/articles` for direct topic-to-article generation and an SSE
streaming variant. A warm vLLM reply takes seconds; a scale-to-zero cold start
was observed at roughly 2–3 minutes, so page entry triggers best-effort warmup
and generation retries transient 502/503/504 responses.

## Challenges, limitations, and next steps

The biggest engineering issue was Modal's cold GPU startup returning 503 before
an upstream existed. Bounded readiness polling, retries, and UI progress make
this recoverable, but cannot remove cold-start latency without a paid warm
container. Another issue was a local dashboard querying its own empty model
volume while the deployed adapter lived on Modal; the UI now says that the
registry is local rather than claiming no adapter exists.

The dataset is small, English-only, single-publisher, and its briefs are
inferred from finished articles. Source-grounded examples were not in the
fine-tuning set. Generated claims still require editorial fact-checking, and
the prototype is browser-scoped rather than production multi-tenant. Next
steps are human review of more held-out topics (especially Thai), a complete
13-pair style adjudication, evidence/citation checks, and a persistent warm
serving tier only if usage justifies its cost. The submission requires a public
prototype URL; that deployment is separate from the verified local flow.
