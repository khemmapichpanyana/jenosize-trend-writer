-- Studio: live training telemetry, agent chats with artifacts, uploaded assets
-- and published (shareable) content.

-- ---------------------------------------------------------------------------
-- training_progress: one row per logged training step, written by the GPU job
-- itself so the console can draw the loss curve and GPU gauges live.
-- ---------------------------------------------------------------------------
create table if not exists public.training_progress (
    id                bigint generated always as identity primary key,
    job_id            uuid not null references public.job_runs (id) on delete cascade,
    created_at        timestamptz not null default now(),
    phase             text not null check (phase in ('loading', 'training', 'saving', 'done', 'failed')),
    step              integer,
    total_steps       integer,
    epoch             real,
    loss              real,
    learning_rate     real,
    grad_norm         real,
    samples_per_sec   real,
    gpu_util          real,           -- percent
    gpu_mem_used_gb   real,
    gpu_mem_total_gb  real,
    message           text
);
create index if not exists training_progress_job_idx on public.training_progress (job_id, id);

-- ---------------------------------------------------------------------------
-- Agent chats. Messages keep only what a person reads (user/assistant text,
-- plus a compact record of tool calls); the agent's scratch state is rebuilt
-- from them each turn.
-- ---------------------------------------------------------------------------
create table if not exists public.chat_threads (
    id          uuid primary key default gen_random_uuid(),
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now(),
    title       text not null default 'New chat'
);
create index if not exists chat_threads_updated_idx on public.chat_threads (updated_at desc);

create table if not exists public.chat_messages (
    id          uuid primary key default gen_random_uuid(),
    thread_id   uuid not null references public.chat_threads (id) on delete cascade,
    created_at  timestamptz not null default now(),
    role        text not null check (role in ('user', 'assistant')),
    content     text not null default '',
    tool_calls  jsonb not null default '[]'::jsonb,   -- [{name, args, summary}]
    asset_ids   uuid[] not null default '{}',         -- images attached to a user turn
    model       text                                  -- which model answered (primary / fallback)
);
create index if not exists chat_messages_thread_idx on public.chat_messages (thread_id, created_at);

-- ---------------------------------------------------------------------------
-- Artifacts: the documents a chat produces. Every edit is a new immutable
-- version, so "what did we publish?" always has an exact answer.
-- ---------------------------------------------------------------------------
create table if not exists public.artifacts (
    id               uuid primary key default gen_random_uuid(),
    thread_id        uuid not null references public.chat_threads (id) on delete cascade,
    created_at       timestamptz not null default now(),
    updated_at       timestamptz not null default now(),
    title            text not null,
    current_version  integer not null default 0
);
create index if not exists artifacts_thread_idx on public.artifacts (thread_id);

create table if not exists public.artifact_versions (
    id           uuid primary key default gen_random_uuid(),
    artifact_id  uuid not null references public.artifacts (id) on delete cascade,
    version      integer not null,
    created_at   timestamptz not null default now(),
    markdown     text,                 -- the article, as written by the fine-tuned model
    html         text,                 -- the designed, sanitised page body (asset:// image refs)
    meta         jsonb not null default '{}'::jsonb,   -- title, meta_description, quality_report, generation_id
    note         text,
    unique (artifact_id, version)
);

-- ---------------------------------------------------------------------------
-- Uploaded images. Private until a published page references them.
-- ---------------------------------------------------------------------------
create table if not exists public.assets (
    id            uuid primary key default gen_random_uuid(),
    created_at    timestamptz not null default now(),
    thread_id     uuid references public.chat_threads (id) on delete set null,
    filename      text not null,
    content_type  text not null,
    bytes         integer not null,
    width         integer,
    height        integer,
    sha256        text not null,
    r2_key        text not null
);
create index if not exists assets_thread_idx on public.assets (thread_id);

-- ---------------------------------------------------------------------------
-- Published pages: a frozen artifact version at a shareable slug.
-- ---------------------------------------------------------------------------
create table if not exists public.published_content (
    id            uuid primary key default gen_random_uuid(),
    created_at    timestamptz not null default now(),
    updated_at    timestamptz not null default now(),
    slug          text not null unique check (slug ~ '^[a-z0-9]+(-[a-z0-9]+)*$'),
    artifact_id   uuid not null references public.artifacts (id) on delete cascade,
    version       integer not null,
    title         text not null,
    r2_key        text not null,
    asset_ids     uuid[] not null default '{}',   -- the only assets made public by this page
    status        text not null default 'published' check (status in ('published', 'unpublished'))
);
create index if not exists published_content_artifact_idx on public.published_content (artifact_id);

alter table public.training_progress  enable row level security;
alter table public.chat_threads       enable row level security;
alter table public.chat_messages      enable row level security;
alter table public.artifacts          enable row level security;
alter table public.artifact_versions  enable row level security;
alter table public.assets             enable row level security;
alter table public.published_content  enable row level security;
