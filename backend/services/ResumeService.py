"""
Resume upload pipeline orchestrator.

Steps (Fix 3):
  1. Extract plain text from the uploaded file (PDF or DOCX).
  2. Parse the text into structured sections via Claude Haiku tool-use.
  3. Replace existing rows in work_experience / projects / education / skills.
  4. Re-read those rows and feed them to the signal extractor (Claude Haiku
     tool-use) to produce the signal_profile.
  5. Persist signal_profile + resume_parsed=true on profiles.
  6. Return parsed sections + signal_profile to the caller.

A separate background coroutine (`run_background_scrape`) is exposed for
the API endpoint to schedule via FastAPI BackgroundTasks once the response
has been sent.
"""

from __future__ import annotations

import asyncio
import logging

from database.database_manager.profiles import ProfileRepository
from database.database_manager.resume_sections import ResumeSectionsRepository
from parsing.resume import (
    ResumeParser,
    SignalExtractor,
    extract_resume_text,
)
from services.JobDiscoveryService import (
    DEFAULT_STREAM_TARGET,
    JobDiscoveryService,
)

logger = logging.getLogger(__name__)


# Module-level singleton: JobDiscoveryService loads sentence-transformers
# and the Anthropic client at construction time. We only want to pay that
# cost once per process, even if multiple resumes get uploaded back-to-back.
_DISCOVERY = JobDiscoveryService(top_n=20)


class ResumeProcessingError(RuntimeError):
    """Raised by ResumeService when a step fails. Caller maps to HTTP 4xx/5xx."""


class ResumeService:
    """High-level orchestration. Construct once and reuse — internals are
    stateless beyond their LLM clients."""

    def __init__(self) -> None:
        self._parser = ResumeParser()
        self._signal = SignalExtractor()
        self._sections = ResumeSectionsRepository()
        self._profiles = ProfileRepository()

    @staticmethod
    def _derive_search_keywords(signal_profile: dict) -> list[str]:
        """
        Build a resilient keyword list from signal_profile.

        Preference order:
          1) search_keywords (if present)
          2) primary_roles
          3) core_skills
          4) domain_expertise

        Dedupes case-insensitively and caps to 6 to match the extractor
        contract.
        """
        if not isinstance(signal_profile, dict):
            return []

        ordered: list[str] = []
        for field in (
            "search_keywords",
            "primary_roles",
            "core_skills",
            "domain_expertise",
        ):
            values = signal_profile.get(field) or []
            if not isinstance(values, list):
                continue
            for raw in values:
                if not isinstance(raw, str):
                    continue
                v = raw.strip()
                if v:
                    ordered.append(v)

        out: list[str] = []
        seen: set[str] = set()
        for kw in ordered:
            key = kw.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(kw)
            if len(out) >= 6:
                break
        return out

    # ------------------------------------------------------------------
    # Foreground pipeline (steps 1–5). Each step fails loud — no silent
    # fallbacks. Caller must convert ResumeProcessingError to a 4xx/5xx.
    # ------------------------------------------------------------------

    async def process_upload(
        self,
        *,
        user_id: str,
        file_bytes: bytes,
        filename: str | None,
        content_type: str | None,
    ) -> dict:
        if not user_id:
            raise ResumeProcessingError("user_id is required")

        # Step 1 — extract text. Sync; tiny.
        try:
            text = await asyncio.to_thread(
                extract_resume_text,
                data=file_bytes,
                filename=filename,
                content_type=content_type,
            )
        except Exception as exc:
            logger.exception("resume text extraction failed user_id=%s", user_id)
            raise ResumeProcessingError(str(exc)) from exc

        # Step 2 — parse to structured sections.
        try:
            parsed = await asyncio.to_thread(self._parser.parse, text)
        except Exception as exc:
            logger.exception("resume_parser failed user_id=%s", user_id)
            raise ResumeProcessingError(
                f"Resume could not be parsed: {exc}"
            ) from exc

        # Step 3 — delete-then-insert into the four section tables.
        try:
            inserted_counts = await asyncio.to_thread(
                self._replace_all_sections, user_id, parsed
            )
        except Exception as exc:
            logger.exception("section persistence failed user_id=%s", user_id)
            raise ResumeProcessingError(
                f"Could not save parsed resume sections: {exc}"
            ) from exc

        # Step 4 — re-read and run signal extraction on what's actually in DB.
        # (Re-reading rather than reusing `parsed` keeps signal_profile in
        # sync with what the user can later see/edit in their profile pages.)
        try:
            sections_in_db = await asyncio.to_thread(
                self._sections.get_all_for_user, user_id
            )
        except Exception as exc:
            logger.exception("section read-back failed user_id=%s", user_id)
            raise ResumeProcessingError(
                f"Could not read back saved sections: {exc}"
            ) from exc

        try:
            signal_profile = await asyncio.to_thread(
                self._signal.extract, sections_in_db
            )
        except Exception as exc:
            logger.exception("signal extraction failed user_id=%s", user_id)
            raise ResumeProcessingError(
                f"Signal extraction failed: {exc}"
            ) from exc

        # Step 5 — flip resume_parsed and persist signal_profile atomically.
        try:
            await asyncio.to_thread(
                self._profiles.update_signal_profile, user_id, signal_profile
            )
        except Exception as exc:
            logger.exception("update_signal_profile failed user_id=%s", user_id)
            raise ResumeProcessingError(
                f"Could not save your signal profile: {exc}"
            ) from exc

        return {
            "sections": sections_in_db,
            "signal_profile": signal_profile,
            "inserted_counts": inserted_counts,
        }

    # ------------------------------------------------------------------
    # Step 6 — background scrape (run after response is sent).
    # ------------------------------------------------------------------

    async def run_background_scrape(
        self,
        *,
        user_id: str,
        signal_profile: dict | None = None,
        search_keywords: list[str] | None = None,
        scrape_target_n: int = 30,
        stream_target: int = DEFAULT_STREAM_TARGET,
        lock_pre_acquired: bool = False,
    ) -> None:
        """
        Streaming background discovery: scrape -> verify -> insert as 'shown'
        on a rolling basis. We never query scrapelist as a candidate source
        (per product spec — scrapelist is for the durable record + the
        already-seen lookup only).

        When ``lock_pre_acquired`` is True the caller (typically the resume
        upload endpoint) has already flipped scraping_in_progress=true so
        the lock is continuously held from the moment the response is sent
        until this task finishes. That stops a racing /api/jobs/queue poll
        from triggering a second concurrent run.

        When False, the task acquires the lock itself and bails if it can't
        (another in-flight scrape is already running for the same user).

        Errors are swallowed + logged — the upload response has already
        been sent, so we cannot surface failures to the user. The frontend's
        polling timeout shows a generic retry message in that case.
        """
        try:
            # Reload signal_profile if not supplied. The upload endpoint
            # passes it inline; the carousel "find more" path may not.
            if signal_profile is None:
                signal_profile = await asyncio.to_thread(
                    self._profiles.get_signal_profile, user_id
                )
            if not signal_profile:
                logger.warning(
                    "Background scrape skipped — no signal_profile for user_id=%s",
                    user_id,
                )
                return

            if search_keywords is None:
                search_keywords = self._derive_search_keywords(signal_profile)
            if not search_keywords:
                logger.warning(
                    "Background scrape skipped — no usable keyterms could be "
                    "derived from signal_profile for user_id=%s",
                    user_id,
                )
                return
            # Echo through to signal_profile so downstream logging is consistent.
            signal_profile = {**signal_profile, "search_keywords": search_keywords}

            if not lock_pre_acquired:
                acquired = await asyncio.to_thread(
                    self._profiles.try_acquire_scraping_lock, user_id
                )
                if not acquired:
                    logger.warning(
                        "Background scrape skipped — scraping_in_progress "
                        "already held for user_id=%s",
                        user_id,
                    )
                    return

            try:
                inserted = await _DISCOVERY.discover_via_streaming(
                    user_id=user_id,
                    signal_profile=signal_profile,
                    scrape_target_n=scrape_target_n,
                    stream_target=stream_target,
                )
                logger.info(
                    "Background scrape done — inserted %d shown rows for user_id=%s",
                    inserted,
                    user_id,
                )
            except Exception:
                logger.exception(
                    "Background scrape failed for user_id=%s — frontend "
                    "polling will time out and surface a retry message.",
                    user_id,
                )
        finally:
            # Always release. Whether we pre-acquired or self-acquired, by
            # the time we exit this task the user is done waiting on us.
            await asyncio.to_thread(
                self._profiles.release_scraping_lock, user_id
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _replace_all_sections(self, user_id: str, parsed: dict) -> dict:
        """Run the four delete-then-insert calls. Returns counts inserted."""
        return {
            "work_experience": self._sections.replace_work_experience(
                user_id, parsed.get("work_experience") or []
            ),
            "projects": self._sections.replace_projects(
                user_id, parsed.get("projects") or []
            ),
            "education": self._sections.replace_education(
                user_id, parsed.get("education") or []
            ),
            "skills": self._sections.replace_skills(
                user_id, parsed.get("skills") or []
            ),
        }
