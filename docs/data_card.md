# Data Card — Jenosize Trend Writer training corpus

> **Status: template.** `make dataset VERSION=v1` renders the real card, with
> actual counts, distributions and the dataset fingerprint, and publishes it to
> R2 as `datasets/v1/data_card.md`. Copy it over this file once v1 is final.

## Dataset summary

| Field | Value |
|---|---|
| Name | `jenosize-ideas-{version}` |
| Version | TBD |
| Built on | TBD |
| Task | Conditional article generation (brief → Jenosize-style article) |
| Language(s) | English (Thai subset TBD) |
| Size | TBD examples (TBD train / TBD eval) |
| Format | JSONL, OpenAI chat messages |

## Provenance

| Field | Value |
|---|---|
| Source | Public articles from Jenosize Ideas (`jenosize.com`) |
| Collection method | Sitemap discovery → polite crawl (~1 req/s, robots.txt respected) |
| Raw archive | Cloudflare R2, `raw/scrape/{yyyy-mm-dd}/{sha256}.html` |
| Collection window | TBD |
| Licence / rights | Company-owned content used for an internal take-home assignment. Not redistributed; only the LoRA adapter is published. |

## Construction

1. **Scrape** (`pipeline/scrape.py`) — archive raw HTML, dedupe on sha256.
2. **Clean** (`pipeline/clean.py`) — boilerplate removal, heading normalisation
   to `##`, byline/date stripping, near-duplicate removal, length filter.
3. **Reverse-label** (`pipeline/label.py`) — an OpenAI-compatible model infers
   the brief (topic, category, industry, audience, keywords, length) that each
   article answers. The labeller sees only the article.
4. **Build** (`pipeline/build_dataset.py`) — render each pair through the *live*
   `app/services/prompt.py` functions so the training prompt is byte-identical
   to the inference prompt, then split deterministically by URL hash.

## Fields

| Field | Description |
|---|---|
| `messages[0]` (system) | Style rules + output contract from `prompt.build_system_prompt` |
| `messages[1]` (user) | Rendered brief from `prompt.build_user_prompt` |
| `messages[2]` (assistant) | `TITLE: … / META: … / --- / markdown body` |

## Splits

| Split | Count | Selection |
|---|---|---|
| train | TBD | deterministic hash of URL |
| eval | TBD | held out; never seen during training |

## Statistics (TBD)

* Word count: min / median / max
* Sections per article: median
* Industry distribution
* Category distribution
* Duplicate rate before/after dedupe

## Known limitations

* **Single-source corpus.** Everything comes from one publisher, so the model
  learns that house voice and nothing else — which is the goal, but it will not
  generalise to another brand's tone.
* **Inferred labels.** Briefs are reconstructed, not authored. A label can be a
  plausible-but-wrong reading of the article; `pipeline/label.py audit` exists
  to sample and read them.
* **Small corpus.** A few hundred articles is enough for style, not for facts —
  which is why retrieval, not the adapter, supplies the evidence.
* **Recency.** The corpus is frozen at collection time; trends age.
* **English-dominant.** Thai output is supported at the API level but thinly
  represented in training data.

## Ethical notes

* No personal data is collected; articles are corporate publications.
* Author names and bylines are stripped so the model cannot attribute text to a
  real person.
* The published adapter is style-only; it does not memorise the corpus verbatim
  (verified at eval time by n-gram overlap against the training set — TBD).
