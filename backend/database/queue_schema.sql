-- Additive migration for the carousel-driven job queue.
-- Safe to run multiple times. Does not touch existing rows.

-- 1. Per-user ordering, fit narrative, action timestamp on user_job_interactions.
alter table public.user_job_interactions
    add column if not exists shown_order integer,
    add column if not exists fit_reason  text,
    add column if not exists acted_at    timestamptz;

create index if not exists user_job_interactions_user_status_idx
    on public.user_job_interactions (user_id, status);

create index if not exists user_job_interactions_user_shown_order_idx
    on public.user_job_interactions (user_id, shown_order)
    where status = 'shown';

-- 2. profiles needs the signal vector + an idempotency lock for scraping.
alter table public.profiles
    add column if not exists signal_profile        jsonb,
    add column if not exists scraping_in_progress  boolean not null default false;
