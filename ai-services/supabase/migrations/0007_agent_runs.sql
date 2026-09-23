-- Agent turns run as Modal background workers instead of inside an HTTP request.
--
-- Sending a message creates an agent_run and returns at once; a worker container
-- executes the turn and appends every event (tokens, tool steps, article drafts)
-- to agent_events. The console follows /v1/studio/runs/{id}/events and resumes
-- from the last event id after a reload or a dropped connection, so closing the
-- tab no longer stops the agent.

create table if not exists public.agent_runs (
    id             uuid primary key default gen_random_uuid(),
    thread_id      uuid not null references public.chat_threads (id) on delete cascade,
    message_id     uuid references public.chat_messages (id) on delete set null,  -- the user turn
    status         text not null default 'queued'
                   check (status in ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    modal_call_id  text,
    error          text,
    created_at     timestamptz not null default now(),
    started_at     timestamptz,
    finished_at    timestamptz
);

-- One turn at a time per chat, enforced by the database: two concurrent turns
-- would interleave replies and race to version the same artifact.
create unique index if not exists agent_runs_one_active_per_thread
    on public.agent_runs (thread_id) where status in ('queued', 'running');
create index if not exists agent_runs_thread_idx on public.agent_runs (thread_id, created_at desc);

-- The run's event log. Consecutive token / artifact_delta events are coalesced
-- by the worker (~4 writes/s), so a 1,000-word article is dozens of rows, not
-- thousands.
create table if not exists public.agent_events (
    id          bigint generated always as identity primary key,
    run_id      uuid not null references public.agent_runs (id) on delete cascade,
    created_at  timestamptz not null default now(),
    type        text not null,
    data        jsonb not null default '{}'::jsonb
);
create index if not exists agent_events_run_idx on public.agent_events (run_id, id);

alter table public.agent_runs   enable row level security;
alter table public.agent_events enable row level security;
