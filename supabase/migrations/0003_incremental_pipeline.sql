-- Incremental pipeline: only new or changed articles cause writes.
--
-- Change detection is by *content fingerprint*, not by raw bytes or sitemap
-- dates, because both of those were measured to be useless on jenosize.com:
--   * every <lastmod> in the sitemap is the site's build time, identical for
--     all 174 articles;
--   * the raw HTML differs on every request (Cloudflare rewrites its
--     email-protection tokens per response), and there is no ETag or
--     Last-Modified header to use instead.
-- The extracted article text, however, is byte-stable across fetches, so its
-- sha256 is the fingerprint.
--
-- Each downstream stage records the fingerprint it last processed. A stage has
-- work to do exactly when its recorded fingerprint differs from the current
-- one, so re-running the whole pipeline is always safe and always cheap.

comment on column public.training_articles.content_hash is
    'sha256 of the extracted article text (NOT the raw HTML, which changes per request).';

alter table public.training_articles
    add column if not exists first_seen_at        timestamptz not null default now(),
    add column if not exists last_checked_at      timestamptz,
    add column if not exists content_changed_at   timestamptz,
    add column if not exists fetch_count          integer not null default 0,
    -- clean stage: which content (and which version of the cleaning rules) the
    -- stored clean_markdown was produced from.
    add column if not exists cleaned_hash         text,
    add column if not exists clean_version        integer,
    -- label stage: labels cost LLM calls, so they are keyed on the content only
    -- (not the cleaner version): tweaking a cleaning rule must not re-bill.
    add column if not exists labelled_hash        text,
    add column if not exists label_version        integer,
    add column if not exists labeler_model        text;

create index if not exists training_articles_last_checked_idx
    on public.training_articles (last_checked_at nulls first);

-- ---------------------------------------------------------------------------
-- pipeline_runs: one row per stage invocation. Makes "what did last night's run
-- actually do?" a query instead of a log search.
-- ---------------------------------------------------------------------------
create table if not exists public.pipeline_runs (
    id           uuid primary key default gen_random_uuid(),
    stage        text not null check (stage in ('discover', 'crawl', 'clean', 'label', 'build')),
    status       text not null default 'running' check (status in ('running', 'succeeded', 'failed')),
    started_at   timestamptz not null default now(),
    finished_at  timestamptz,
    stats        jsonb not null default '{}'::jsonb,
    error        text
);

create index if not exists pipeline_runs_started_idx on public.pipeline_runs (started_at desc);

-- ---------------------------------------------------------------------------
-- dataset_versions: training sets are immutable once published. A model card
-- that says "trained on v1" must mean the same bytes forever, so re-building v1
-- from a changed corpus is refused rather than silently overwritten.
-- ---------------------------------------------------------------------------
create table if not exists public.dataset_versions (
    version      text primary key,
    created_at   timestamptz not null default now(),
    fingerprint  text not null,             -- sha256 over train.jsonl + eval.jsonl
    train_count  integer not null,
    eval_count   integer not null,
    train_key    text not null,
    eval_key     text not null,
    card_key     text not null,
    params       jsonb not null default '{}'::jsonb
);

alter table public.pipeline_runs    enable row level security;
alter table public.dataset_versions enable row level security;
