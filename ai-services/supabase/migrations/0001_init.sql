-- Jenosize Trend Writer — initial schema.
--
-- Security posture: the product has NO user auth. Nothing should ever reach
-- Postgres from a browser. Therefore every table has RLS enabled with zero
-- policies, which denies the anon and authenticated roles outright; the API
-- connects with the service/secret key, which bypasses RLS.
--
-- Style: uuid PKs, timestamptz defaults, jsonb for anything whose shape is
-- still moving (requests, quality reports, metrics).

create extension if not exists "pgcrypto";  -- gen_random_uuid()

-- ---------------------------------------------------------------------------
-- model_versions: which adapter produced which article.
-- ---------------------------------------------------------------------------
create table if not exists public.model_versions (
    id                uuid primary key default gen_random_uuid(),
    created_at        timestamptz not null default now(),
    name              text not null unique,           -- e.g. 'jeno-lora-v1'
    base_model        text not null,                  -- e.g. 'Qwen/Qwen3-4B-Instruct-2507'
    adapter_uri       text,                           -- HF repo or Modal volume path
    serving_endpoint  text,                           -- Modal web_server URL
    is_active         boolean not null default false,
    metrics           jsonb not null default '{}'::jsonb
);

-- Only one adapter may be live at a time; enforced in the DB so a bad deploy
-- cannot leave two "active" versions and make results unattributable.
create unique index if not exists model_versions_single_active
    on public.model_versions (is_active) where is_active;

-- ---------------------------------------------------------------------------
-- source_documents: everything retrieval can draw on.
-- ---------------------------------------------------------------------------
create table if not exists public.source_documents (
    id            uuid primary key default gen_random_uuid(),
    created_at    timestamptz not null default now(),
    kind          text not null check (kind in ('url', 'upload', 'scrape')),
    url           text,
    r2_key        text,                                -- archived original bytes
    content_hash  text unique,                         -- sha256; makes re-ingest idempotent
    title         text,
    text_chars    integer not null default 0,
    metadata      jsonb not null default '{}'::jsonb,  -- carries the extracted text
    fetched_at    timestamptz not null default now()
);

create index if not exists source_documents_kind_idx on public.source_documents (kind);
create index if not exists source_documents_url_idx on public.source_documents (url);

-- ---------------------------------------------------------------------------
-- generations: one row per article request, from queued to terminal state.
-- ---------------------------------------------------------------------------
create table if not exists public.generations (
    id                uuid primary key default gen_random_uuid(),
    created_at        timestamptz not null default now(),
    status            text not null default 'queued'
                      check (status in ('queued', 'running', 'succeeded', 'failed')),
    request           jsonb not null default '{}'::jsonb,   -- verbatim ArticleRequest
    normalized_params jsonb not null default '{}'::jsonb,   -- what the prompt actually used
    model_version_id  uuid references public.model_versions (id) on delete set null,
    title             text,
    meta_description  text,
    article_markdown  text,
    r2_key            text,
    quality_report    jsonb,
    sources           jsonb not null default '[]'::jsonb,
    latency_ms        integer,
    prompt_tokens     integer,
    completion_tokens integer,
    error             text,
    completed_at      timestamptz
);

-- The two access patterns the API has: newest-first listing, and status sweeps
-- for rows stranded in 'running' by a lambda timeout.
create index if not exists generations_created_at_idx on public.generations (created_at desc);
create index if not exists generations_status_idx on public.generations (status);

-- ---------------------------------------------------------------------------
-- generation_sources: which chunk of which document fed which article.
-- Kept separate from generations.sources (a denormalised snapshot) because this
-- is the table that answers "is retrieval actually helping?" at eval time.
-- ---------------------------------------------------------------------------
create table if not exists public.generation_sources (
    id                  uuid primary key default gen_random_uuid(),
    created_at          timestamptz not null default now(),
    generation_id       uuid not null references public.generations (id) on delete cascade,
    source_document_id  uuid references public.source_documents (id) on delete set null,
    chunk_index         integer not null default 0,
    score               real not null default 0,
    snippet             text
);

create index if not exists generation_sources_generation_idx
    on public.generation_sources (generation_id);

-- ---------------------------------------------------------------------------
-- training_articles: the fine-tuning corpus (Day 1 pipeline writes here).
-- ---------------------------------------------------------------------------
create table if not exists public.training_articles (
    id             uuid primary key default gen_random_uuid(),
    created_at     timestamptz not null default now(),
    url            text not null unique,
    title          text,
    r2_raw_key     text,                                 -- archived raw HTML
    clean_markdown text,
    word_count     integer not null default 0,
    labels         jsonb not null default '{}'::jsonb,   -- reverse-labelled brief
    split          text not null default 'train' check (split in ('train', 'eval'))
);

create index if not exists training_articles_split_idx on public.training_articles (split);

-- ---------------------------------------------------------------------------
-- eval_runs / eval_results: base vs fine-tuned comparisons (Day 2).
-- ---------------------------------------------------------------------------
create table if not exists public.eval_runs (
    id                uuid primary key default gen_random_uuid(),
    created_at        timestamptz not null default now(),
    model_version_id  uuid references public.model_versions (id) on delete set null,
    name              text not null,
    summary           jsonb not null default '{}'::jsonb
);

create table if not exists public.eval_results (
    id           uuid primary key default gen_random_uuid(),
    created_at   timestamptz not null default now(),
    eval_run_id  uuid not null references public.eval_runs (id) on delete cascade,
    model_label  text not null,                        -- 'base' | 'finetuned' | adapter name
    prompt       jsonb not null default '{}'::jsonb,
    output_md    text,
    scores       jsonb not null default '{}'::jsonb
);

create index if not exists eval_results_run_idx on public.eval_results (eval_run_id);

-- ---------------------------------------------------------------------------
-- RLS: enabled everywhere, no policies anywhere. Deny by default.
-- ---------------------------------------------------------------------------
alter table public.model_versions     enable row level security;
alter table public.source_documents   enable row level security;
alter table public.generations        enable row level security;
alter table public.generation_sources enable row level security;
alter table public.training_articles  enable row level security;
alter table public.eval_runs          enable row level security;
alter table public.eval_results       enable row level security;
