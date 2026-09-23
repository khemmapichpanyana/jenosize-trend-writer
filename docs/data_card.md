# Data card — Jenosize Ideas v2

Generated on 2026-09-21 by `pipeline/build_dataset.py`. The exact immutable
card is included with `train.jsonl` and `eval.jsonl` in the private submission
bundle and stored as `datasets/v2/data_card.md` in R2.

| Field | Value |
|---|---|
| Task | Conditional English business-trend article generation |
| Source | Public Jenosize Ideas articles, discovered through sitemaps |
| Collection | ~1 request/second, robots.txt respected; raw HTML archived in R2 |
| Accepted corpus | 159 cleaned and reverse-labelled articles |
| Quality-filtered v2 | 105 examples: 92 train, 13 held-out eval |
| Split | Deterministic SHA-256 of source URL, seed 13, eval fraction 0.1 |
| Fingerprint | `69f0b4147fd86e1e3cfadb5c7e6d2db1c716b3f32e1b9c960af002bc2d864e4c` |
| Format | JSONL: system prompt, rendered user brief, article response |
| Labeller | `gpt-5.6-luna` reverse-labelled topic, industry, audience, and keywords |

## Pipeline

`pipeline/scrape.py` archives and deduplicates source pages.
`pipeline/clean.py` removes navigation, CTAs, bylines, dates, and reference
lists, normalizes headings, rejects 300–4000 word outliers, and marks near
duplicates. `pipeline/label.py` reconstructs briefs from the finished article;
category and language are derived rather than guessed. `pipeline/build_dataset.py`
renders each pair through the live `app/services/prompt.py` functions, validates
the output contract, and writes an immutable train/eval split.

The included 105 examples have 548–1,603 words (median 839). Largest categories
are Transformation and Technology (28), Real-time Marketing (26), and
Understand People and Consumer (19). The v2 quality filter excluded 54
candidates: 9 failed title length, 12 meta length, and 43 had fewer than three
H2 sections; categories overlap.

## Limitations and use

The corpus is English, single-publisher, and relatively small. Its briefs are
inferred, not original editor instructions. Training examples have no supplied
source-document blocks, so retrieval-grounding is carried by the base model
and inference prompt, not learned by the adapter. Articles can become stale;
generated factual claims need editorial review. Bylines and dates are stripped
to avoid attribution to real authors. This dataset and adapter are for private
assignment review, not public redistribution.
