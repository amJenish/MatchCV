"""
Carousel job queue endpoints.

Behaviour after the rolling-discovery refactor:

* GET  /api/jobs/queue       — pure read of the user's current 'shown' rows.
                               No discovery is triggered here. The frontend
                               polls this while a background scrape is in
                               progress (kicked off by /api/resume/upload or
                               /api/jobs/find-more) and re-renders as new
                               rows appear.
* POST /api/jobs/interact    — set status=applied|not_interested|saved.
* POST /api/jobs/find-more   — kicks off a streaming scrape+insert in the
                               background and returns immediately. Frontend
                               then polls /queue. Discovery does NOT query
                               scrapelist for candidates — scrapelist is
                               strictly the durable record + the lookup
                               table for already-seen jobs.
* GET  /api/jobs/applied     — read-only list of applied jobs.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.auth import require_user
from api.jobs import _row_to_item  # reuse the same response shape
from database.database_manager.profiles import ProfileRepository
from database.database_manager.user_jobs import (
    CAP,
    UserJobInteractionsRepository,
)
from services.ResumeService import ResumeService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/jobs",
    tags=["queue"],
    dependencies=[Depends(require_user)],
)

# Module-level singletons. ResumeService owns the JobDiscoveryService
# instance so the heavy sentence-transformers / Anthropic init happens
# exactly once per process.
_resume_service = ResumeService()
_user_jobs = UserJobInteractionsRepository()
_profiles = ProfileRepository()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


def _serialize_card(row: dict) -> dict:
    """Map a user_job_interactions+scrapelist join row to a frontend card."""
    job_row = row.get("job") or {}
    job_view = _row_to_item(job_row).model_dump() if job_row else {}
    return {
        "interaction_id": row.get("id"),
        "shown_order": row.get("shown_order"),
        "fit_reason": row.get("fit_reason"),
        "fit_score": row.get("fit_score"),
        "composite_score": row.get("composite_score"),
        "legitimacy_score": row.get("legitimacy_score") or job_view.get("legitimacy_score"),
        "quality_score": row.get("quality_score") or job_view.get("quality_score"),
        "job": job_view,
    }


def _serialize_applied(row: dict) -> dict:
    job_row = row.get("job") or {}
    job_view = _row_to_item(job_row).model_dump() if job_row else {}
    return {
        "interaction_id": row.get("id"),
        "fit_reason": row.get("fit_reason"),
        "fit_score": row.get("fit_score"),
        "acted_at": row.get("acted_at"),
        "created_at": row.get("created_at"),
        "job": job_view,
    }


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class InteractBody(BaseModel):
    interaction_id: str
    status: str = Field(..., pattern="^(applied|not_interested|saved)$")


# ---------------------------------------------------------------------------
# GET /api/jobs/queue — pure read
# ---------------------------------------------------------------------------

@router.get("/queue")
async def get_queue(user: Any = Depends(require_user)) -> JSONResponse:
    user_id = str(user.id)

    rows = await asyncio.to_thread(_user_jobs.list_shown_with_jobs, user_id)
    shown_count = len(rows)
    discovery_in_progress = await asyncio.to_thread(
        _profiles.is_scraping_in_progress, user_id
    )

    return JSONResponse(
        content={
            "shown_count": shown_count,
            "cap_reached": shown_count >= CAP,
            "available_slots": max(CAP - shown_count, 0),
            "discovery_in_progress": discovery_in_progress,
            "jobs": [_serialize_card(r) for r in rows],
        }
    )


# ---------------------------------------------------------------------------
# POST /api/jobs/interact
# ---------------------------------------------------------------------------

@router.post("/interact")
async def post_interact(
    body: InteractBody,
    user: Any = Depends(require_user),
) -> JSONResponse:
    user_id = str(user.id)

    try:
        updated = await asyncio.to_thread(
            _user_jobs.update_status,
            body.interaction_id,
            user_id,
            body.status,
        )
    except ValueError as exc:
        return _error(400, str(exc))
    except Exception:
        logger.exception("interact update failed user_id=%s", user_id)
        return _error(502, "Could not update interaction")

    if not updated:
        return _error(404, "Interaction not found")

    shown_count = await asyncio.to_thread(_user_jobs.count_shown, user_id)
    return JSONResponse(
        content={
            "interaction_id": updated.get("id"),
            "status": updated.get("status"),
            "shown_count": shown_count,
            "cap_reached": shown_count >= CAP,
            "available_slots": max(CAP - shown_count, 0),
        }
    )


# ---------------------------------------------------------------------------
# POST /api/jobs/find-more — fire-and-forget background streaming scrape
# ---------------------------------------------------------------------------

@router.post("/find-more")
async def post_find_more(
    background_tasks: BackgroundTasks,
    user: Any = Depends(require_user),
) -> JSONResponse:
    user_id = str(user.id)

    shown_count = await asyncio.to_thread(_user_jobs.count_shown, user_id)
    if shown_count >= CAP:
        return _error(
            400,
            f"You already have {CAP} jobs in your queue. Apply or dismiss "
            "some before fetching more.",
        )

    signal_profile = await asyncio.to_thread(
        _profiles.get_signal_profile, user_id
    )
    if not signal_profile:
        return _error(
            400,
            "Upload your resume to build your signal profile before "
            "we can fetch more jobs.",
        )
    keywords = _resume_service._derive_search_keywords(signal_profile)
    if not keywords:
        return _error(
            400,
            "Your signal profile has no search keywords. Re-upload your "
            "resume so we can rebuild it.",
        )

    # Pre-acquire the lock here so the immediate /queue poll the frontend
    # fires off after this 200 sees discovery_in_progress=true. The bg
    # task is responsible for the release.
    lock_held = await asyncio.to_thread(
        _profiles.try_acquire_scraping_lock, user_id
    )
    if not lock_held:
        return _error(409, "A job-fetch run is already in progress.")

    background_tasks.add_task(
        _resume_service.run_background_scrape,
        user_id=user_id,
        signal_profile=signal_profile,
        search_keywords=keywords,
        lock_pre_acquired=True,
    )

    return JSONResponse(
        content={
            "shown_count": shown_count,
            "cap_reached": False,
            "available_slots": max(CAP - shown_count, 0),
            "discovery_in_progress": True,
            "message": (
                "Searching for more jobs in the background. Poll the "
                "queue endpoint to see new ones as they arrive."
            ),
        }
    )


# ---------------------------------------------------------------------------
# GET /api/jobs/applied
# ---------------------------------------------------------------------------

@router.get("/applied")
async def get_applied(user: Any = Depends(require_user)) -> JSONResponse:
    user_id = str(user.id)
    rows = await asyncio.to_thread(_user_jobs.list_applied_with_jobs, user_id)
    return JSONResponse(
        content={
            "count": len(rows),
            "jobs": [_serialize_applied(r) for r in rows],
        }
    )
