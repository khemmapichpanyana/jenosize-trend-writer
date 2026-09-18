-- Columns the data pipeline needs on `training_articles`.
--
-- Added as a second migration rather than by editing 0001 so the applied
-- history stays honest. Everything here is nullable with a default, so it is
-- safe to run against a table that already holds rows.
--
-- Why each column exists:
--   category_slug    the publisher's own taxonomy, taken from the URL path.
--                    Free, accurate `category` labels that need no LLM guess.
--   meta_description the page's own <meta> description. Ground truth for the
--                    `META:` line the model is trained to emit, rather than a
--                    description we invented.
--   content_hash     sha256 of the raw HTML; makes a re-crawl idempotent.
--   is_duplicate     set by the shingle-Jaccard pass in pipeline/clean.py.
--                    Flagged, not deleted, so the dedupe decision is auditable.
--   error            why an article dropped out (fetch failed, too short,
--                    unparseable label). Keeps failures visible instead of
--                    silently shrinking the corpus.
--   fetched_at       when the raw HTML was captured; the corpus is a snapshot
--                    and the data card has to say as of when.

alter table public.training_articles
    add column if not exists category_slug    text,
    add column if not exists meta_description text,
    add column if not exists content_hash     text,
    add column if not exists is_duplicate     boolean not null default false,
    add column if not exists error            text,
    add column if not exists fetched_at       timestamptz;

-- The dataset builder only ever selects usable rows, and on a few hundred
-- articles this is the one query shape worth an index.
create index if not exists training_articles_usable_idx
    on public.training_articles (split)
    where clean_markdown is not null and not is_duplicate;

create index if not exists training_articles_category_idx
    on public.training_articles (category_slug);

-- 0001 already enabled RLS on this table with no policies; new columns inherit
-- that, so nothing further is needed here.
