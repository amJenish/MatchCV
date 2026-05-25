"""
Resume upload — POST /api/resume/upload (Fix 3).

Step 1–5 run in the request lifecycle and return the parsed sections +
signal profile to the client. Step 6 (ScrapingService) is queued via
FastAPI BackgroundTasks so the user sees the response immediately and
the frontend starts polling /api/jobs/queue.

Lock handoff: this endpoint pre-acquires profiles.scraping_in_progress
*before* returning so a racing /queue poll cannot run discovery against
a stale scrapelist while the background scrape is still warming up.
The background task then releases the lock when done.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    UploadFile,
)
from fastapi.responses import JSONResponse

from api.auth import require_user
from database.database_manager.profiles import ProfileRepository
from services.ResumeService import ResumeProcessingError, ResumeService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/resume",
    tags=["resume"],
    dependencies=[Depends(require_user)],
)

# Module-level singletons — Anthropic clients are reused across uploads.
_resume_service = ResumeService()
_profiles = ProfileRepository()

# Hard cap so a malicious / oversized upload doesn't OOM the worker.
MAX_BYTES = 6 * 1024 * 1024  # 6 MiB

ALLOWED_EXTENSIONS = (".pdf", ".docx")


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


@router.post("/upload")
async def upload_resume(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user: Any = Depends(require_user),
) -> JSONResponse:
    user_id = str(user.id)

    # Cheap pre-check — saves an Anthropic call when the upload is wrong.
    name = (file.filename or "").lower().strip()
    if not name.endswith(ALLOWED_EXTENSIONS):
        return _error(400, "Unsupported file type. Upload a PDF or DOCX resume.")

    try:
        body = await file.read()
    except Exception:
        logger.exception("upload read failed user_id=%s", user_id)
        return _error(400, "Could not read the uploaded file.")
    finally:
        await file.close()

    if not body:
        return _error(400, "Uploaded file is empty.")
    if len(body) > MAX_BYTES:
        return _error(
            400, f"File is too large. Max size is {MAX_BYTES // (1024 * 1024)} MB."
        )

    try:
        result = await _resume_service.process_upload(
            user_id=user_id,
            file_bytes=body,
            filename=file.filename,
            content_type=file.content_type,
        )
    except ResumeProcessingError as exc:
        return _error(400, str(exc))
    except Exception:
        logger.exception("resume upload pipeline crashed user_id=%s", user_id)
        return _error(500, "Resume processing failed unexpectedly.")

    signal_profile = result.get("signal_profile") or {}
    keywords = signal_profile.get("search_keywords") or []

    # Pre-acquire the scraping lock so any /queue poll between now and the
    # bg task starting up sees discovery_in_progress=true (and waits)
    # instead of racing into discovery against a stale scrapelist.
    lock_held = await asyncio.to_thread(
        _profiles.try_acquire_scraping_lock, user_id
    )

    # Schedule step 6 (rolling streaming discovery) — runs after the
    # response is delivered. We pass the freshly-built signal_profile
    # inline so the bg task doesn't have to round-trip the DB to find it.
    background_tasks.add_task(
        _resume_service.run_background_scrape,
        user_id=user_id,
        signal_profile=signal_profile,
        search_keywords=list(keywords),
        lock_pre_acquired=lock_held,
    )

    return JSONResponse(
        content={
            "sections": result.get("sections"),
            "signal_profile": signal_profile,
            "inserted_counts": result.get("inserted_counts"),
            "message": (
                "Resume parsed. We are finding jobs for you in the "
                "background — this may take a moment."
            ),
        }
    )
