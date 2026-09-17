# Jenosize Trend Writer — Technical Report

> **Status: outline.** Sections marked _(Day 1)_ / _(Day 2)_ are filled in as the
> data pipeline and fine-tuning land. Written against the grading split:
> fine-tuning 40%, data engineering 20%, deployment 20%, documentation 20%.

---

## 1. Problem statement

- What the service produces: business trend / future-ideas articles in the
  Jenosize Ideas voice.
- Inputs: topic, category, industry, target audience, SEO keywords, optional
  source URL or uploaded document.
- What "good" means here: on-voice, structured, SEO-complete, factually
  restrained.

## 2. Approach

- **Core thesis: fine-tuning teaches style, retrieval supplies facts.**
- Why a 4B base + LoRA rather than prompting a frontier model: cost per article,
  latency, and the fact that the deliverable is a *fine-tune*.
- Why retrieval is separate: a small model that invents statistics is worse than
  no model; grounding is a data problem, not a weights problem.
- What is deliberately *not* done: no RLHF, no agent loop, no vector DB.

## 3. Data engineering _(Day 1)_

- Corpus: source, size, collection method, dedupe strategy.
- Cleaning rules and what each one prevents the model from learning.
- **Reverse-labelling**: inferring the brief from the finished article, and the
  two rules that keep labels honest (labeller sees only the article; every
  inferred industry is normalised with the shipped `normalize_industry`).
- Prompt-parity invariant: the dataset is rendered through the same
  `app/services/prompt.py` functions the API calls at inference.
- Split strategy and leakage checks.
- → see `docs/data_card.md`.

## 4. Fine-tuning _(Day 2)_

- Base model: `Qwen/Qwen3-4B-Instruct-2507`. Why this one (size, licence,
  instruction quality, Thai coverage).
- Method: Unsloth LoRA, r=16, alpha=16, lr 2e-4, 3 epochs, max_seq_len 4096,
  4-bit base.
- Loss masking on the assistant turn only, and why it matters.
- Hardware, wall-clock and cost per run (L4 on Modal).
- Hyperparameter choices and what was tried.
- Failure modes observed and how they were diagnosed.

## 5. Evaluation _(Day 2)_

- Deterministic metrics (`app/services/quality.py`): word count, `##` count,
  keyword coverage, title length — cheap, reproducible, gate the API.
- Pairwise LLM judge on style adherence: base vs fine-tuned, held-out briefs.
- Memorisation check: n-gram overlap with the training corpus.
- Results table + honest discussion of where the fine-tune does *not* help.

## 6. Serving and deployment

- FastAPI on Vercel (CPU) + vLLM on Modal (GPU), and why the split is forced by
  Vercel's 500 MB Python bundle limit.
- Scale-to-zero economics; cold starts; `POST /model/warmup`; SSE heartbeats.
- Pluggable seams (model / persistence / storage) and the zero-account local
  mode that makes the service reproducible by a grader.
- Error model, structured logging, request ids.
- → see `docs/architecture.md`.

## 7. Results

- Public endpoint and demo instructions.
- Latency profile: cold vs warm, blocking vs streaming.
- Example generations, with the quality report shown alongside.

## 8. Limitations and next steps

- Single-publisher corpus; inferred labels; English-dominant.
- Retrieval is BM25 over per-request documents, not a persistent index.
- No background job queue: generation happens inside the request.
- Next: larger corpus, embedding retrieval over a persistent index, Thai
  evaluation set, human review loop on quality-gate failures.

## 9. Appendix

- Repository layout.
- Environment variables.
- Reproduction steps (train → eval → deploy).
