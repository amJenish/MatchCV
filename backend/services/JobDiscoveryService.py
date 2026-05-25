import asyncio
import heapq
import logging
from dataclasses import dataclass, field

from database.database_manager.profiles import ProfileRepository
from database.database_manager.scrapelist import ScrapelistRepository
from database.database_manager.user_jobs import UserJobInteractionsRepository
from parsing.constants import QUEUE_END
from validating.job_fit_checker import JobFitChecker
from validating.job_fit_prefilter import (
    ensure_job_embeddings,
    prefilter_top_n,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TOP_N = 10
DEFAULT_PREFILTER_N = 40
DEFAULT_CANDIDATE_LIMIT = 500
DEFAULT_FIT_CONCURRENCY = 6

# Display-time gates. A job only enters the user's carousel when ALL three
# strict-greater-than thresholds are cleared:
#   legitimacy_score > 74  (passes at 75+)
#   quality_score    > 79  (passes at 80+)
#   fit_score        > 50  (passes at 51+; fit_score is total_score 0-100
#                           from JobFitChecker)
HIGH_QUALITY_LEGITIMACY_MIN = 74
HIGH_QUALITY_QUALITY_MIN    = 79
MIN_FIT_SCORE               = 50
DEFAULT_STREAM_TARGET       = 5
DEFAULT_QUEUE_CAP           = 20


# ---------------------------------------------------------------------------
# Signal-profile <-> profile-shape adapter
#
# The cheap prefilter and the LLM fit-checker were originally written
# against the resume sections shape (work_experience / projects /
# education). Fix 5 says discovery must source its scoring from the
# signal_profile fields instead. Rather than refactor those modules —
# which work — we inflate the signal profile into a dict that mirrors
# the shape they expect, distributing the signal across the existing
# work / project / education weights.
# ---------------------------------------------------------------------------

# JobScraper's enum is lower-case; signal_profile uses these display strings.
SENIORITY_MAP: dict[str, list[str]] = {
    "Intern":    ["intern"],
    "Junior":    ["junior"],
    "Mid-level": ["mid"],
    # Senior is an umbrella for all senior+ scrapelist tiers — staff,
    # principal, and people-managers all overlap a senior IC's skill set.
    "Senior":    ["senior", "staff", "principal", "manager"],
}


def _seniority_filter(suitable_levels) -> list[str] | None:
    if not suitable_levels or not isinstance(suitable_levels, list):
        return None
    out: list[str] = []
    for lvl in suitable_levels:
        for mapped in SENIORITY_MAP.get(str(lvl), []):
            if mapped not in out:
                out.append(mapped)
    return out or None


def signal_profile_to_prefilter_shape(signal_profile: dict) -> dict:
    """
    Map a signal_profile (search_keywords / core_skills / primary_roles /
    secondary_roles / domain_expertise) into the work/projects/education
    shape that JobFitPrefilter and JobFitChecker already consume.

    Distribution:
      work       ← primary_roles  + core_skills + domain_expertise (highest weight)
      projects   ← search_keywords + core_skills                   (medium)
      education  ← secondary_roles + domain_expertise              (lowest)
    """
    primary   = list(signal_profile.get("primary_roles") or [])
    secondary = list(signal_profile.get("secondary_roles") or [])
    skills    = list(signal_profile.get("core_skills") or [])
    domain    = list(signal_profile.get("domain_expertise") or [])
    keywords  = list(signal_profile.get("search_keywords") or [])

    work = [
        {
            "work_title": role,
            "company": "",
            "work_info": skills + domain,
        }
        for role in primary or [""]
    ]
    projects = [
        {
            "project_title": kw,
            "project_info": [],
            "tech_stack": skills,
        }
        for kw in (keywords or [""])
    ]
    education = [
        {
            "credential_name":  role,
            "institution":      "",
            "field_of_study":   " ".join(domain),
        }
        for role in secondary
    ] or [{"credential_name": "", "institution": "", "field_of_study": " ".join(domain)}]

    return {
        # JobFitChecker may also surface these directly in its prompt.
        "search_keywords":  keywords,
        "primary_roles":    primary,
        "secondary_roles":  secondary,
        "core_skills":      skills,
        "domain_expertise": domain,
        "suitable_levels":  signal_profile.get("suitable_levels") or [],
        "deal_breakers":    signal_profile.get("deal_breakers") or [],
        "years_of_experience": signal_profile.get("years_of_experience"),
        # Shape compatibility with the existing prefilter / fit checker.
        "work_experience":  work,
        "projects":         projects,
        "education":        education,
    }


# ---------------------------------------------------------------------------
# Job slot — one entry in the priority queue
# ---------------------------------------------------------------------------

@dataclass
class JobSlot:
    score: float
    job:   dict = field(compare=False)

    def __lt__(self, other: "JobSlot") -> bool:
        # Higher score = higher priority.
        return self.score > other.score

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "job":   self.job,
        }


# ---------------------------------------------------------------------------
# Priority queue (max-heap of size C)
# ---------------------------------------------------------------------------

class JobPriorityQueue:
    """
    Keeps the best C jobs seen so far. Eviction is greedy: a new job
    replaces the current worst only if it scores higher.
    """

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("Capacity must be at least 1.")
        self.capacity = capacity
        self._heap: list[JobSlot] = []   # min-heap on score

    def add(self, job: dict, score: float) -> bool:
        slot = JobSlot(score=score, job=job)
        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, slot)
            return True
        worst = self._heap[0]
        if score > worst.score:
            heapq.heapreplace(self._heap, slot)
            return True
        return False

    def is_full(self) -> bool:
        return len(self._heap) >= self.capacity

    def size(self) -> int:
        return len(self._heap)

    def worst_score(self) -> float | None:
        return self._heap[0].score if self._heap else None

    def results(self) -> list[dict]:
        return [slot.to_dict() for slot in sorted(self._heap, reverse=True)]

    def __len__(self) -> int:
        return len(self._heap)

    def __repr__(self) -> str:
        return (
            f"JobPriorityQueue(capacity={self.capacity}, "
            f"size={self.size()}, worst={self.worst_score()})"
        )


# ---------------------------------------------------------------------------
# Discovery service
# ---------------------------------------------------------------------------

class JobDiscoveryService:
    """
    End-to-end discovery for a single user click ("show my top N jobs").

    Pipeline
    --------
    1. Pull recent scrapelist rows passing legitimacy/quality gates that
       the user has not interacted with yet.
    2. Lazy-fill `job_embedding` for any rows missing it (single batched
       OpenAI call) and persist back to scrapelist for future queries.
    3. Prefilter to `prefilter_n` (cheap keyword + embedding cosine, no LLM).
    4. Run JobFitChecker rubric (Anthropic Haiku) in parallel on the
       survivors with bounded concurrency.
    5. Composite score = 50% fit + 30% legitimacy + 20% quality, fed into
       a max-heap of size `top_n`.

    Parameters
    ----------
    top_n : int
        How many final jobs to surface (priority queue capacity).
    prefilter_n : int
        How many candidates survive the cheap prefilter and get LLM-scored.
    candidate_limit : int
        Max rows to pull from scrapelist per query.
    fit_concurrency : int
        Max in-flight Anthropic JobFitChecker calls.
    """

    FIT_WEIGHT        = 0.50
    LEGITIMACY_WEIGHT = 0.30
    QUALITY_WEIGHT    = 0.20

    def __init__(
        self,
        top_n: int = DEFAULT_TOP_N,
        *,
        prefilter_n: int = DEFAULT_PREFILTER_N,
        candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
        fit_concurrency: int = DEFAULT_FIT_CONCURRENCY,
    ):
        if top_n < 1:
            raise ValueError("top_n must be at least 1.")
        if prefilter_n < top_n:
            raise ValueError("prefilter_n must be >= top_n.")

        self.top_n = top_n
        self.prefilter_n = prefilter_n
        self.candidate_limit = candidate_limit
        self.fit_concurrency = fit_concurrency

        self._scrapelist = ScrapelistRepository()
        self._profiles = ProfileRepository()
        self._user_jobs = UserJobInteractionsRepository()
        self._fit_checker = JobFitChecker()
        self.queue = JobPriorityQueue(capacity=top_n)
        # Backwards-compat alias for older callers that referenced .c.
        self.c = top_n

        logger.info(
            "JobDiscoveryService initialised — top_n=%d prefilter_n=%d",
            top_n,
            prefilter_n,
        )

    # ------------------------------------------------------------------
    # Composite scoring
    # ------------------------------------------------------------------

    def compute_score(
        self,
        fit_score:        float,
        legitimacy_score: float,
        quality_score:    float,
    ) -> float:
        fit        = max(0.0, min(1.0, fit_score))
        legitimacy = max(0.0, min(1.0, legitimacy_score))
        quality    = max(0.0, min(1.0, quality_score))
        return (
            fit        * self.FIT_WEIGHT
            + legitimacy * self.LEGITIMACY_WEIGHT
            + quality    * self.QUALITY_WEIGHT
        )

    def add_job(
        self,
        job:              dict,
        fit_score:        float,
        legitimacy_score: float,
        quality_score:    float,
    ) -> bool:
        score = self.compute_score(fit_score, legitimacy_score, quality_score)
        return self.queue.add(job=job, score=score)

    def is_complete(self) -> bool:
        return self.queue.is_full()

    def results(self) -> list[dict]:
        return self.queue.results()

    # ------------------------------------------------------------------
    # Main entry — call this on "show my top N"
    # ------------------------------------------------------------------

    async def search_existing_jobs(
        self,
        user_id: str,
        *,
        cap: int | None = None,
        signal_profile: dict | None = None,
    ) -> list[dict]:
        """
        Returns the user's top-N composite-scored jobs from scrapelist.
        Output: list of {"score": float, "job": dict} sorted best -> worst.
        Each job dict has `fit_result` and `prefilter_score` attached.

        Pipeline (Fix 5):
          1. Load signal_profile from public.profiles when not supplied.
             Raises ValueError if still null — discovery must not run
             against an unparsed user.
          2. Pull unseen candidates via the NOT-EXISTS RPC, applying
             suitable_levels seniority pre-filter at the SQL layer.
          3. Lazy-fill job embeddings + persist back to scrapelist.
          4. Score with the cheap prefilter (keyword + cosine on the
             search_keywords / core_skills / primary_roles signal).
          5. Run JobFitChecker rubric in bounded parallel.
          6. Composite-score into a max-heap of size effective_top_n.

        `cap` clamps the result count to `min(self.top_n, cap)`. Used by
        /api/jobs/queue to respect available_slots. Cap of 0 returns [].
        """
        if not user_id:
            raise ValueError("user_id is required for discovery")
        if cap is not None and cap <= 0:
            return []
        effective_top_n = min(self.top_n, cap) if cap is not None else self.top_n
        self.queue = JobPriorityQueue(capacity=effective_top_n)

        # Step 1 — signal profile is the matching source.
        if signal_profile is None:
            signal_profile = await asyncio.to_thread(
                self._profiles.get_signal_profile, user_id
            )
        if not signal_profile:
            raise ValueError(
                "JobDiscoveryService: signal_profile is missing — the user "
                "must upload a resume before discovery can run."
            )

        seniority_levels = _seniority_filter(signal_profile.get("suitable_levels"))
        adapter = signal_profile_to_prefilter_shape(signal_profile)

        # Step 2 — single-query unseen fetch with seniority pre-filter.
        candidates = await asyncio.to_thread(
            self._scrapelist.fetch_unseen_for_user_signal,
            user_id,
            seniority_levels=seniority_levels,
            limit=self.candidate_limit,
        )
        logger.info(
            "search_existing_jobs: %d candidates (seniority=%s) for user=%s",
            len(candidates),
            seniority_levels,
            user_id,
        )
        if not candidates:
            return []

        # Step 3 — lazy fill embeddings + persist for next time.
        to_persist = await ensure_job_embeddings(candidates)
        if to_persist:
            logger.info(
                "search_existing_jobs: persisting %d new job embeddings",
                len(to_persist),
            )
            await asyncio.to_thread(
                self._scrapelist.bulk_update_job_embeddings, to_persist
            )

        # Step 4 — cheap prefilter against the signal-derived adapter shape.
        top = await prefilter_top_n(adapter, candidates, n=self.prefilter_n)
        logger.info(
            "search_existing_jobs: prefilter kept %d / %d",
            len(top),
            len(candidates),
        )
        if not top:
            return []

        # Step 5 — LLM rubric in bounded parallel.
        fit_results = await self._run_fit_checks([j for j, _ in top], adapter)

        # Step 6 — composite + heap.
        for (job, prefilter_score), fit in zip(top, fit_results):
            job["fit_result"] = fit
            job["prefilter_score"] = round(prefilter_score, 4)
            self.add_job(
                job=job,
                fit_score=(fit.get("total_score") or 0) / 100.0,
                legitimacy_score=(job.get("legitimacy_score") or 0) / 100.0,
                quality_score=(job.get("quality_score") or 0) / 100.0,
            )

        out = self.results()
        logger.info(
            "search_existing_jobs: returning %d / %d", len(out), effective_top_n
        )
        return out

    # ------------------------------------------------------------------
    # Streaming discovery — consume verified jobs off ScrapingService.run
    # ------------------------------------------------------------------

    @staticmethod
    def _format_fit_reason(fit_result: dict) -> str | None:
        if not fit_result:
            return None
        for key in ("summary", "explanation", "rationale", "reason"):
            value = fit_result.get(key)
            if value and isinstance(value, str):
                return value.strip()
        components = fit_result.get("components") or {}
        if isinstance(components, dict):
            notes = [c.get("explanation") for c in components.values() if isinstance(c, dict)]
            notes = [n for n in notes if n]
            if notes:
                return notes[0].strip()
        return None

    async def stream_from_scraper(
        self,
        *,
        user_id: str,
        signal_profile: dict,
        result_queue: asyncio.Queue,
        target: int = DEFAULT_STREAM_TARGET,
        cap: int = DEFAULT_QUEUE_CAP,
        stop_event: asyncio.Event | None = None,
    ) -> int:
        """
        Pull verified jobs off ScrapingService.run's result_queue as they
        arrive. Each job that clears the high-quality bar
        (legitimacy_score > 74 AND quality_score > 80), matches the user's
        seniority preference, and isn't already in user_job_interactions
        gets a fit-check + immediate insert as 'shown'.

        Inserts are written one-at-a-time so the user's carousel grows on
        a rolling basis as the scraper produces material. When `target`
        new rows have been inserted (or the per-user cap of 20 is hit) we
        signal `stop_event` so the scraper can stop early and skip the
        remaining LLM round-trips.

        Returns the number of new 'shown' rows inserted in this run.
        """
        if not user_id:
            raise ValueError("user_id is required")
        if not signal_profile:
            raise ValueError("signal_profile is required (no fallback to scrapelist)")

        seniority_levels = _seniority_filter(signal_profile.get("suitable_levels"))
        adapter = signal_profile_to_prefilter_shape(signal_profile)

        current_shown = await asyncio.to_thread(self._user_jobs.count_shown, user_id)
        slots_left = max(0, cap - current_shown)
        if slots_left <= 0:
            if stop_event is not None:
                stop_event.set()
            return 0

        target_this_run = min(target, slots_left)
        next_order = await asyncio.to_thread(self._user_jobs.max_shown_order, user_id)

        inserted = 0
        rejected_quality = 0
        rejected_seniority = 0
        rejected_seen = 0
        rejected_fit = 0

        while True:
            job = await result_queue.get()
            try:
                if job is QUEUE_END:
                    break

                # 1. High-quality gate (legitimacy > 74 AND quality > 79).
                lscore = int(job.get("legitimacy_score") or 0)
                qscore = int(job.get("quality_score") or 0)
                if (
                    lscore <= HIGH_QUALITY_LEGITIMACY_MIN
                    or qscore <= HIGH_QUALITY_QUALITY_MIN
                ):
                    rejected_quality += 1
                    logger.info(
                        "stream: drop %s — l=%d q=%d below high-quality bar",
                        job.get("job_title"),
                        lscore,
                        qscore,
                    )
                    continue

                # 2. Seniority filter (NULL passes through, mismatch drops).
                job_level = job.get("seniority_level")
                if seniority_levels and job_level:
                    if str(job_level).lower() not in seniority_levels:
                        rejected_seniority += 1
                        continue

                # 3. Resolve scrapelist id (set by ScrapingService persist step).
                job_id = job.get("id")
                if not job_id:
                    posting_url = job.get("posting_url")
                    if posting_url:
                        job_id = await asyncio.to_thread(
                            self._scrapelist.lookup_id_by_posting_url, posting_url
                        )
                if not job_id:
                    logger.warning(
                        "stream: dropping %s — no scrapelist id resolved",
                        job.get("posting_url") or job.get("job_title"),
                    )
                    continue

                # 4. Skip if user has any prior interaction with this job.
                if await asyncio.to_thread(
                    self._user_jobs.has_interaction, user_id, job_id
                ):
                    rejected_seen += 1
                    continue

                # 5. Fit check (LLM) + fit gate. fit_total is 0-100 from
                # JobFitChecker; we drop anything <= MIN_FIT_SCORE before
                # paying the per-user insert cost.
                fit_result = await asyncio.to_thread(
                    self._fit_checker.check, job, adapter
                )
                fit_total = float(fit_result.get("total_score") or 0)
                if fit_total <= MIN_FIT_SCORE:
                    rejected_fit += 1
                    logger.info(
                        "stream: drop %s — fit=%d below MIN_FIT_SCORE=%d",
                        job.get("job_title"),
                        int(fit_total),
                        MIN_FIT_SCORE,
                    )
                    continue
                fit_norm = max(0.0, min(1.0, fit_total / 100.0))

                # 6. Insert as 'shown' immediately so the carousel renders it.
                next_order += 1
                composite = self.compute_score(
                    fit_norm,
                    lscore / 100.0,
                    qscore / 100.0,
                )
                payload = {
                    "job_id":           job_id,
                    "fit_reason":       self._format_fit_reason(fit_result),
                    "fit_score":        round(fit_norm, 4),
                    "legitimacy_score": lscore,
                    "quality_score":    qscore,
                    "composite_score":  round(composite, 4),
                }
                row = await asyncio.to_thread(
                    self._user_jobs.insert_shown_one,
                    user_id,
                    payload,
                    next_order,
                )
                if row:
                    inserted += 1
                    logger.info(
                        "stream: inserted shown #%d/%d — %s @ %s "
                        "(l=%d q=%d fit=%d composite=%.3f)",
                        inserted,
                        target_this_run,
                        job.get("job_title"),
                        job.get("company_name"),
                        lscore,
                        qscore,
                        int(fit_total),
                        composite,
                    )
                else:
                    # Conflict — already inserted between has_interaction
                    # and now (rare race). No-op, keep streaming.
                    next_order -= 1

                if inserted >= target_this_run:
                    if stop_event is not None:
                        stop_event.set()
                    break
            finally:
                result_queue.task_done()

        logger.info(
            "stream done — inserted=%d target=%d "
            "(rejected: quality=%d seniority=%d already-seen=%d fit=%d) "
            "user=%s",
            inserted,
            target_this_run,
            rejected_quality,
            rejected_seniority,
            rejected_seen,
            rejected_fit,
            user_id,
        )
        return inserted

    async def discover_via_streaming(
        self,
        *,
        user_id: str,
        signal_profile: dict,
        scrape_target_n: int = 30,
        stream_target: int = DEFAULT_STREAM_TARGET,
    ) -> int:
        """
        End-to-end rolling discovery: scrape -> consume -> insert.

        ScrapingService and the streaming consumer run concurrently so the
        user's carousel fills as fresh jobs are scraped. We never query
        scrapelist as a candidate source — scrapelist is now strictly the
        durable record + the lookup table for already-seen jobs.

        ``scrape_target_n`` is how many prefiltered jobs the scraper
        commits to verifying before giving up. The streaming consumer
        signals early stop the moment ``stream_target`` good rows have
        been inserted, so excess scraping is cheap to bail out of.

        Returns the number of new 'shown' rows inserted for this user.
        """
        # Local import keeps the module-level singleton creation in
        # api/queue.py from cycling back through ScrapingService.
        from services.ScrapingService import ScrapingService

        keywords = list(signal_profile.get("search_keywords") or [])
        if not keywords:
            logger.warning(
                "discover_via_streaming: no search_keywords on signal_profile "
                "for user_id=%s — scraper will fall back to default sources",
                user_id,
            )

        result_queue: asyncio.Queue = asyncio.Queue()
        stop_event = asyncio.Event()

        scraping = ScrapingService(n=scrape_target_n)

        scrape_task = asyncio.create_task(
            scraping.scrape(
                keywords,
                result_queue=result_queue,
                external_stop=stop_event,
            ),
            name=f"scrape-{user_id}",
        )
        consume_task = asyncio.create_task(
            self.stream_from_scraper(
                user_id=user_id,
                signal_profile=signal_profile,
                result_queue=result_queue,
                target=stream_target,
                stop_event=stop_event,
            ),
            name=f"stream-consume-{user_id}",
        )

        try:
            inserted = await consume_task
        except Exception:
            logger.exception(
                "discover_via_streaming: streaming consumer crashed for "
                "user_id=%s",
                user_id,
            )
            stop_event.set()
            inserted = 0

        # Make sure the scraper finishes (drains its tasks, posts QUEUE_END).
        # If the consumer reached its target the scraper will see stop_event
        # and bail out within seconds. Otherwise it runs to natural completion.
        try:
            await scrape_task
        except Exception:
            logger.exception(
                "discover_via_streaming: scraper task raised for user_id=%s",
                user_id,
            )

        return inserted

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _run_fit_checks(
        self,
        jobs: list[dict],
        profile: dict,
    ) -> list[dict]:
        """
        Run JobFitChecker.check on each job in parallel via threads,
        bounded by `fit_concurrency`. Anthropic SDK calls are blocking,
        so threads keep the asyncio loop responsive.
        """
        sem = asyncio.Semaphore(self.fit_concurrency)

        async def _one(job: dict) -> dict:
            async with sem:
                return await asyncio.to_thread(
                    self._fit_checker.check, job, profile
                )

        return await asyncio.gather(*[_one(j) for j in jobs])

    def __repr__(self) -> str:
        return (
            f"JobDiscoveryService(top_n={self.top_n}, "
            f"prefilter_n={self.prefilter_n}, "
            f"collected={self.queue.size()})"
        )
