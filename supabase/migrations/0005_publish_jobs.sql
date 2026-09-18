-- Publishing an adapter to Hugging Face becomes a job like the others, so it
-- is started over the jobs API and shows up in job_runs with its outcome.

alter table public.job_runs drop constraint if exists job_runs_kind_check;
alter table public.job_runs
    add constraint job_runs_kind_check
    check (kind in ('scrape', 'label', 'train', 'eval', 'publish'));
