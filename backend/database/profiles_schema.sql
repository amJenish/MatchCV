-- profiles table (auth.users is managed by Supabase)
-- Run in Supabase SQL editor.

create table if not exists public.profiles (
    id          uuid primary key references auth.users(id) on delete cascade,
    full_name   text,
    last_name   text,
    email       text,
    resume_parsed boolean default false,
    created_at  timestamptz not null default now()
);
