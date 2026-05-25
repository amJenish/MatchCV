import asyncio
import logging

from database.database_manager.scrapelist import ScrapelistRepository
from parsing.constants import QUEUE_END
from parsing.extractor import JobExtractor
from parsing.prefilter import prefilter
from parsing.scraper import JobScraper, normalize_url
from validating.job_verification.legitimacy_check import legitimacy_check
from validating.job_verification.quality_check import assess_job_quality

logger = logging.getLogger(__name__)

MIN_EXTRACTION_CONFIDENCE = 0.4
MIN_QUALITY_SCORE = 35


def _build_merged_job(
    scraper_job: dict,
    extracted: dict,
    legitimacy: dict,
    quality: dict,
) -> dict:
    """
    Assemble the final pipeline dict with explicit precedence so that no
    component silently overwrites a peer's field. The extractor is the
    source of truth for posting fields; legitimacy and quality contribute
    only their own keys.
    """
    extracted_domain = extracted.get("company_domain")
    clearbit_domain = (
        (legitimacy.get("company_verification") or {}).get("matched_domain")
    )
    clearbit_logo = (
        (legitimacy.get("company_verification") or {}).get("matched_logo")
    )
    canonical_domain = extracted_domain or clearbit_domain
    canonical_logo = extracted.get("company_logo") or clearbit_logo

    merged = dict(extracted)
    merged["source"] = scraper_job.get("source")
    merged["posting_url"] = scraper_job.get("url") or extracted.get("posting_url")
    merged["content_hash"] = scraper_job.get("_content_hash")
    merged["company_domain"] = canonical_domain
    merged["company_domain_extracted"] = extracted_domain
    merged["company_domain_clearbit"] = clearbit_domain
    merged["company_logo"] = canonical_logo

    merged["legitimacy_status"] = legitimacy.get("status")
    merged["legitimacy_score"] = legitimacy.get("legitimacy_score")
    merged["legitimacy_flags"] = legitimacy.get("flags") or []
    merged["legitimacy_company_verification"] = legitimacy.get("company_verification")
    merged["status"] = legitimacy.get("status")

    merged["quality_score"] = quality.get("quality_score")
    merged["meets_standard"] = bool(quality.get("meets_standard"))
    merged["quality_components"] = quality.get("quality_components") or {}
    merged["quality_penalties"] = quality.get("quality_penalties") or {}
    merged["quality_failures"] = quality.get("quality_failures") or []
    merged["quality_highlights"] = quality.get("quality_highlights") or []
    merged["quality_summary"] = quality.get("quality_summary") or ""

    apply_url = scraper_job.get("apply_url")
    apply_source = scraper_job.get("apply_url_source")
    ext_apply = extracted.get("apply_url")
    if ext_apply:
        ext_apply = normalize_url(str(ext_apply))
    if not apply_url and ext_apply:
        apply_url = ext_apply
        apply_source = "extractor"
    merged["apply_url"] = apply_url
    merged["apply_url_source"] = apply_source

    return merged


class ScrapingService:
    """
    Scrape -> prefilter -> Extract -> stub-gate -> Legitimacy -> legit-gate
    -> Quality -> quality-gate -> persist -> result_queue.

    Parameters
    ----------
    n : int
        Stop after this many jobs pass prefilter.
    persist : bool
        Whether to insert verified jobs into the scrapelist table.
    """

    def __init__(self, n: int, *, persist: bool = True) -> None:
        if n < 1:
            raise ValueError("n must be at least 1.")
        self.n = n
        self.persist = persist
        self.passed_count = 0
        self.verified_count = 0
        self.inserted_count = 0
        self.output_queue: asyncio.Queue = asyncio.Queue()
        self._extractor = JobExtractor()
        self._scrapelist = ScrapelistRepository()
        logger.info("ScrapingService initialised — collecting %d passing jobs.", n)
        print(
            f"[ScrapingService] Initialised — target prefilter passes: {n}",
            flush=True,
        )

    async def scrape(
        self,
        search_keywords: list[str],
        n: int | None = None,
        *,
        result_queue: asyncio.Queue | None = None,
        external_stop: asyncio.Event | None = None,
    ) -> int:
        """
        Direct scrape API. Caller provides search_keywords explicitly —
        ScrapingService never derives them from profile tables. Used by both
        the resume upload background task and the carousel "Find more" route.

        When `result_queue` is provided the caller can consume verified jobs
        as they are produced (rolling / streaming consumer). Otherwise the
        results are buffered into an internal queue and discarded.

        When `external_stop` is provided the caller can request early
        termination — e.g. once the streaming consumer has accepted the
        target number of jobs.

        Returns the count of jobs that fully passed the verify pipeline.
        """
        if search_keywords is None:
            raise ValueError("search_keywords is required (pass [] for none)")
        if not isinstance(search_keywords, list):
            raise TypeError("search_keywords must be a list of strings")
        if n is not None:
            if n < 1:
                raise ValueError("n must be at least 1.")
            self.n = n

        profile = {"search_keywords": [str(k) for k in search_keywords if k]}
        if result_queue is None:
            result_queue = asyncio.Queue()
        return await self.run(profile, result_queue, external_stop=external_stop)

    async def run(
        self,
        profile: dict,
        result_queue: asyncio.Queue,
        *,
        external_stop: asyncio.Event | None = None,
    ) -> int:
        self.passed_count = 0
        self.verified_count = 0
        self.inserted_count = 0
        work_queue: asyncio.Queue = asyncio.Queue()
        # Two DISTINCT stop signals — do NOT alias them:
        #   stop_scraping  -> internal coordination only (prefilter tells the
        #                     scraper to halt source HTTP fetches once its
        #                     target of `self.n` prefilter passes is reached).
        #                     This signal MUST NOT short-circuit the verifier;
        #                     otherwise the 30 prefiltered jobs would be
        #                     silently drained without any LLM verification.
        #   external_stop  -> from JobDiscoveryService.stream_from_scraper:
        #                     "consumer has its target_this_run inserts, no
        #                     more LLM work needed". The verifier short-
        #                     circuits on this; the scraper also halts via
        #                     the bridge below.
        stop_scraping = asyncio.Event()
        bridge_task: asyncio.Task | None = None
        if external_stop is not None:
            async def _bridge_external_to_internal() -> None:
                # One-way: external_stop flipping by the consumer should also
                # halt the scraper/prefilter. The reverse is intentionally NOT
                # wired so the prefilter hitting `n` doesn't kill the verifier.
                await external_stop.wait()
                stop_scraping.set()
            bridge_task = asyncio.create_task(_bridge_external_to_internal())
        seen_hashes: set[str] = set()

        print(
            "[ScrapingService] run() started — launching scraper, prefilter, and verifier tasks",
            flush=True,
        )

        scraper = JobScraper()
        scrape_task = asyncio.create_task(
            scraper.scrape_into_queue(profile, work_queue, stop_scraping)
        )
        filter_task = asyncio.create_task(
            self._filter_and_forward(
                work_queue, self.output_queue, stop_scraping, seen_hashes
            )
        )
        verify_task = asyncio.create_task(
            self._verify_and_forward(
                self.output_queue, result_queue, external_stop
            )
        )

        print(
            "[ScrapingService] Waiting for scraper + prefilter to finish...",
            flush=True,
        )
        try:
            await asyncio.gather(scrape_task, filter_task)
            print(
                f"[ScrapingService] Scrape/prefilter done — {self.passed_count} passed prefilter; "
                "signalling verifier to finish",
                flush=True,
            )
            await self.output_queue.put(QUEUE_END)
            await verify_task
            await result_queue.put(QUEUE_END)
        finally:
            # Tear down the bridge if external_stop never fired during the run.
            if bridge_task is not None and not bridge_task.done():
                bridge_task.cancel()

        print(
            f"[ScrapingService] run() complete — prefilter passed: {self.passed_count}, "
            f"verified: {self.verified_count}, inserted: {self.inserted_count}",
            flush=True,
        )
        logger.info(
            "ScrapingService complete — %d prefilter passes, %d verified, %d inserted.",
            self.passed_count,
            self.verified_count,
            self.inserted_count,
        )
        return self.verified_count

    async def _filter_and_forward(
        self,
        work_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        stop_scraping: asyncio.Event,
        seen_hashes: set[str],
    ) -> None:
        print(
            "[ScrapingService] Prefilter task started — waiting for scraped jobs",
            flush=True,
        )
        while self.passed_count < self.n and not stop_scraping.is_set():
            job = await work_queue.get()
            try:
                if job is QUEUE_END:
                    print(
                        "[ScrapingService] Prefilter task received QUEUE_END from scraper",
                        flush=True,
                    )
                    break
                if stop_scraping.is_set():
                    # Drain quickly without doing more LLM-heavy work downstream.
                    continue

                if not prefilter(job, seen_hashes=seen_hashes):
                    continue

                await output_queue.put(job)
                self.passed_count += 1
                print(
                    f"[ScrapingService] Prefilter PASS ({self.passed_count}/{self.n}): "
                    f"{job.get('title')} @ {job.get('url')}",
                    flush=True,
                )
                logger.info(
                    "Passed prefilter (%d/%d): %s @ %s",
                    self.passed_count,
                    self.n,
                    job.get("title"),
                    job.get("url"),
                )

                if self.passed_count >= self.n:
                    print(
                        "[ScrapingService] Prefilter target reached — stopping scraper",
                        flush=True,
                    )
                    stop_scraping.set()
                    break
            finally:
                work_queue.task_done()
        print("[ScrapingService] Prefilter task exiting", flush=True)

    async def _verify_and_forward(
        self,
        output_queue: asyncio.Queue,
        result_queue: asyncio.Queue,
        external_stop: asyncio.Event | None,
    ) -> None:
        """
        Drains output_queue, runs each job through extractor + legitimacy +
        quality, persists to scrapelist, and forwards to result_queue.

        external_stop is the streaming consumer's "I have my target inserts,
        skip remaining LLM work" signal. It is intentionally NOT the same
        event the prefilter uses to halt the scraper — that event firing
        must not silently drain prefiltered jobs without verification.
        """
        print(
            "[ScrapingService] Verifier task started — waiting for prefiltered jobs",
            flush=True,
        )
        while True:
            job = await output_queue.get()
            try:
                if job is QUEUE_END:
                    print(
                        "[ScrapingService] Verifier task received QUEUE_END",
                        flush=True,
                    )
                    break
                if external_stop is not None and external_stop.is_set():
                    # Consumer already inserted its target_this_run rows; no
                    # value in paying more LLM cost. Drain quickly to QUEUE_END.
                    continue

                print(
                    f"[ScrapingService] Verifier processing: {job.get('title')} @ {job.get('url')}",
                    flush=True,
                )
                enriched = await self._verify_job(job)
                if enriched is None:
                    continue

                await result_queue.put(enriched)
                self.verified_count += 1
                print(
                    f"[ScrapingService] Verifier DONE ({self.verified_count}): "
                    f"{enriched.get('job_title')} — status={enriched.get('status')} "
                    f"q={enriched.get('quality_score')} l={enriched.get('legitimacy_score')}",
                    flush=True,
                )
                logger.info(
                    "Verified job (%d): %s @ %s",
                    self.verified_count,
                    enriched.get("job_title"),
                    enriched.get("posting_url"),
                )
            finally:
                output_queue.task_done()
        print("[ScrapingService] Verifier task exiting", flush=True)

    async def _verify_job(self, scraper_job: dict) -> dict | None:
        title = scraper_job.get("title") or "?"
        raw_text = scraper_job.get("description") or ""
        posting_url = scraper_job.get("url")

        # 1. Extract.
        print(f"[ScrapingService]   → JobExtractor.extract: {title}", flush=True)
        extracted = await asyncio.to_thread(
            self._extractor.extract, raw_text, posting_url
        )
        print(f"[ScrapingService]   ← JobExtractor.extract: {title}", flush=True)

        if scraper_job.get("apply_url"):
            extracted["apply_url"] = scraper_job.get("apply_url")
            extracted["apply_url_source"] = scraper_job.get("apply_url_source")

        # 2. Stub-gate.
        confidence = float(extracted.get("extraction_confidence") or 0.0)
        if extracted.get("is_truncated_or_stub") or confidence < MIN_EXTRACTION_CONFIDENCE:
            print(
                f"[ScrapingService]   ✗ Dropped at stub-gate: {title} "
                f"(stub={extracted.get('is_truncated_or_stub')}, conf={confidence})",
                flush=True,
            )
            return None

        # 3. Legitimacy on the extracted dict so domain/email/salary are available.
        print(f"[ScrapingService]   → legitimacy_check: {title}", flush=True)
        legitimacy = await legitimacy_check(extracted, scraper_job=scraper_job)
        print(
            f"[ScrapingService]   ← legitimacy_check: {title} — "
            f"status={legitimacy.get('status')} score={legitimacy.get('legitimacy_score')}",
            flush=True,
        )

        # 4. Legit-gate.
        if legitimacy.get("status") == "fraud":
            print(
                f"[ScrapingService]   ✗ Dropped as fraud: {title} — "
                f"flags={[f.get('message') for f in (legitimacy.get('flags') or [])]}",
                flush=True,
            )
            logger.info(
                "Dropped fraud: %s @ %s — %s",
                scraper_job.get("title"),
                scraper_job.get("url"),
                legitimacy.get("flags"),
            )
            return None

        # 5. Quality.
        print(f"[ScrapingService]   → assess_job_quality: {title}", flush=True)
        quality = await assess_job_quality(extracted)
        print(
            f"[ScrapingService]   ← assess_job_quality: {title} — "
            f"score={quality.get('quality_score')} "
            f"meets_standard={quality.get('meets_standard')}",
            flush=True,
        )

        # 6. Quality-gate.
        q_score = int(quality.get("quality_score") or 0)
        if q_score < MIN_QUALITY_SCORE:
            print(
                f"[ScrapingService]   ✗ Dropped at quality-gate: {title} "
                f"(score={q_score} < {MIN_QUALITY_SCORE})",
                flush=True,
            )
            return None

        # 7. Merge + persist.
        merged = _build_merged_job(scraper_job, extracted, legitimacy, quality)
        print(f"[ScrapingService]   ✓ Full job JSON built: {title}", flush=True)

        if self.persist:
            await self._persist_to_scrapelist(merged)

        return merged

    async def _persist_to_scrapelist(self, merged: dict) -> None:
        title = merged.get("job_title") or merged.get("company_name") or "?"
        print(f"[ScrapingService]   → insert scrapelist: {title}", flush=True)
        inserted = await asyncio.to_thread(
            self._scrapelist.insert_verified_job, merged
        )
        if inserted:
            self.inserted_count += 1
            merged["id"] = inserted.get("id")
            print(
                f"[ScrapingService]   ← scrapelist insert OK id={inserted.get('id')} "
                f"— {merged.get('posting_url')}",
                flush=True,
            )
            return

        # Duplicate insert (posting_url or content_hash collision).
        # Fetch the existing row's id so the streaming consumer can still
        # link a user_job_interactions row to it.
        existing_id = await asyncio.to_thread(
            self._scrapelist.lookup_id_by_posting_url, merged.get("posting_url")
        )
        if existing_id:
            merged["id"] = existing_id
            print(
                f"[ScrapingService]   ← scrapelist insert skipped (duplicate); "
                f"linked to existing id={existing_id} — {merged.get('posting_url')}",
                flush=True,
            )
        else:
            print(
                "[ScrapingService]   ← scrapelist insert skipped "
                "(duplicate posting_url, duplicate content_hash, or missing identifier); "
                "no existing id resolved",
                flush=True,
            )

    def __repr__(self) -> str:
        return (
            f"ScrapingService(n={self.n}, "
            f"passed={self.passed_count}, verified={self.verified_count}, "
            f"inserted={self.inserted_count})"
        )
