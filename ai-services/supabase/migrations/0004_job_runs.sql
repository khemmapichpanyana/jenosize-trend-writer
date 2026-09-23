-- job_runs: one row per job started through the jobs API (scrape, label,
-- train, eval). A job can span several pipeline stages, e.g. a scrape job runs
-- discover -> crawl -> clean, and each stage still logs its own pipeline_runs
-- row, now linked back to the job that started it.

create table if not exists public.job_runs (
    id             uuid primary key default gen_random_uuid(),
    kind           text not null check (kind in ('scrape', 'label', 'train', 'eval')),
    status         text not null default 'queued'
                   check (status in ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    params         jsonb not null default '{}'::jsonb,
    result         jsonb,
    error          text,
    modal_call_id  text,                  -- to poll or cancel the underlying Modal call
    created_at     timestamptz not null default now(),
    started_at     timestamptz,
    finished_at    timestamptz
);

-- At most one active job per kind, enforced by the database rather than by a
-- check-then-insert in application code (which two simultaneous requests would
-- both pass). Two concurrent crawls would double-fetch every page; two
-- concurrent trainings would race to write the same adapter directory.
create unique index if not exists job_runs_one_active_per_kind
    on public.job_runs (kind) where status in ('queued', 'running');

create index if not exists job_runs_created_idx on public.job_runs (created_at desc);

alter table public.pipeline_runs
    add column if not exists job_id uuid references public.job_runs (id) on delete set null;

create index if not exists pipeline_runs_job_idx on public.pipeline_runs (job_id);

alter table public.job_runs enable row level security;
