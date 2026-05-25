-- Single-query NOT EXISTS lookup of unseen, gate-passing scrapelist rows
-- for a given user. Replaces the earlier two-query "fetch seen ids then
-- filter" pattern so concurrent inserts can't sneak past discovery.
--
-- Seniority filter:
--   - If p_seniority_levels is NULL → no filter (all rows allowed)
--   - Else jobs with NULL seniority_level pass through (we don't drop
--     unknown-level postings; that would lose too many candidates)
--   - Else the job's seniority_level must match one of the requested levels

create or replace function public.scrapelist_unseen_for_user(
    p_user_id           uuid,
    p_min_legitimacy    int     default 50,
    p_min_quality       int     default 35,
    p_seniority_levels  text[]  default null,
    p_limit             int     default 500
)
returns setof public.scrapelist
language sql
stable
as $$
    select s.*
    from public.scrapelist s
    where s.legitimacy_score >= p_min_legitimacy
      and s.quality_score    >= p_min_quality
      and (
          p_seniority_levels is null
          or s.seniority_level is null
          or lower(s.seniority_level) = any (
              select lower(x) from unnest(p_seniority_levels) as t(x)
          )
      )
      and not exists (
          select 1 from public.user_job_interactions u
          where u.user_id = p_user_id
            and u.job_id  = s.id
      )
    order by s.scraped_at desc
    limit p_limit;
$$;
