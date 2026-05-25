-- Resume sections — populated on every upload.
-- Each row scoped by user_id; the upload flow deletes-then-inserts so
-- we never accumulate stale entries from older resumes.
--
-- NOTE: All four tables already exist in the live database with a
-- specific shape. The CREATE TABLE statements below are `if not exists`
-- so they are no-ops on the live DB. They're kept for fresh installs and
-- as documentation of the canonical column types our code targets.

create table if not exists public.work_experience (
    id          uuid primary key default gen_random_uuid(),
    user_id     uuid not null references auth.users(id) on delete cascade,
    work_title  text not null,
    company     text not null,
    work_start  date not null,
    work_end    date,
    is_current  boolean not null default false,
    work_info   text[]  not null default '{}',
    created_at  timestamptz not null default now()
);
create index if not exists work_experience_user_idx on public.work_experience(user_id);

create table if not exists public.projects (
    id            uuid primary key default gen_random_uuid(),
    user_id       uuid not null references auth.users(id) on delete cascade,
    project_title text,
    project_info  text[]  not null default '{}',
    tech_stack    text[]  not null default '{}',
    created_at    timestamptz not null default now()
);
create index if not exists projects_user_idx on public.projects(user_id);

create table if not exists public.education (
    id               uuid primary key default gen_random_uuid(),
    user_id          uuid not null references auth.users(id) on delete cascade,
    institution      text,
    degree           text,
    field_of_study   text,
    graduation_year  integer,
    created_at       timestamptz not null default now()
);
create index if not exists education_user_idx on public.education(user_id);

create table if not exists public.skills (
    id          uuid primary key default gen_random_uuid(),
    user_id     uuid not null references auth.users(id) on delete cascade,
    skills      text[]  not null default '{}',
    created_at  timestamptz not null default now()
);
create index if not exists skills_user_idx on public.skills(user_id);
