"""GET /api/jobs — list verified jobs from scrapelist for the frontend."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.auth import require_user
from database.database_manager._client import get_client

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["jobs"],
    dependencies=[Depends(require_user)],
)


# ---------------------------------------------------------------------------
# Salary helpers
# ---------------------------------------------------------------------------

# Maps the scrapelist `salary_period` enum to user-facing labels.
# The schema enum is hourly/weekly/monthly/yearly/contract; the front-end
# names mentioned in the spec ("annual", "contract_rate") are also accepted
# defensively.
_PERIOD_LABEL = {
    "yearly": "per year",
    "annual": "per year",
    "monthly": "per month",
    "weekly": "per week",
    "hourly": "per hour",
    "contract": "contract",
    "contract_rate": "contract",
}


def _format_amount(amount: Any) -> str | None:
    """salary_min/max are stored as text; format to a clean digit string when possible."""
    if amount is None:
        return None
    raw = str(amount).strip()
    if not raw:
        return None
    try:
        as_int = int(raw.replace(",", ""))
        return f"{as_int:,}"
    except ValueError:
        return raw


def _compensation_label(period: str | None) -> str | None:
    if not period:
        return None
    return _PERIOD_LABEL.get(period.strip().lower(), period)


def _build_salary_view(row: dict) -> dict:
    """
    Returns either {"salary": "..."} or {"salary_range": {"min": ..., "max": ...}}
    plus salary_currency and compensation_type. Single field if min/max are
    equal or only one exists; range when both exist and differ.
    """
    salary_min = _format_amount(row.get("salary_min"))
    salary_max = _format_amount(row.get("salary_max"))
    currency = row.get("salary_currency") or None
    period_label = _compensation_label(row.get("salary_period"))

    salary: str | None = None
    salary_range: dict[str, str] | None = None

    if salary_min and salary_max:
        if salary_min == salary_max:
            salary = salary_min
        else:
            salary_range = {"min": salary_min, "max": salary_max}
    elif salary_min:
        salary = salary_min
    elif salary_max:
        salary = salary_max

    return {
        "salary": salary,
        "salary_range": salary_range,
        "salary_currency": currency,
        "compensation_type": period_label,
    }


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if x is not None]
    return []


def _remote_label(is_remote: Any) -> str | None:
    if is_remote is True:
        return "Remote"
    if is_remote is False:
        return "On-site"
    return None


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class SalaryRange(BaseModel):
    min: str
    max: str


class JobItem(BaseModel):
    id: str | None = None

    job_title: str | None = None
    company_name: str | None = None
    company_hq_country: str | None = None
    contact_email: str | None = None

    job_description: str | None = None
    responsibilities: list[str] = Field(default_factory=list)
    required_qualifications: list[str] = Field(default_factory=list)
    preferred_qualifications: list[str] = Field(default_factory=list)
    benefits: list[str] = Field(default_factory=list)

    seniority_level: str | None = None
    remote_policy: bool | None = None
    remote_label: str | None = None
    remote_region: str | None = None
    timezone_requirements: str | None = None
    work_authorization: str | None = None

    salary: str | None = None
    salary_range: SalaryRange | None = None
    salary_currency: str | None = None
    compensation_type: str | None = None
    equity_offered: bool = False

    legitimacy_company_verification: dict | None = None
    quality_summary: str | None = None
    legitimacy_score: int | None = None
    quality_score: int | None = None

    apply_url: str | None = None
    apply_url_source: str | None = None
    posting_url: str | None = None


class JobsResponse(BaseModel):
    items: list[JobItem]
    count: int
    limit: int
    offset: int
    legitimacy_score_min: int
    quality_score_min: int


# ---------------------------------------------------------------------------
# Row → response mapper
# ---------------------------------------------------------------------------

def _row_to_item(row: dict) -> JobItem:
    salary_view = _build_salary_view(row)
    salary_range = (
        SalaryRange(**salary_view["salary_range"])
        if salary_view["salary_range"]
        else None
    )

    apply_url = row.get("apply_url") or row.get("posting_url")

    return JobItem(
        id=row.get("id"),
        job_title=row.get("job_title"),
        company_name=row.get("company_name"),
        company_hq_country=row.get("company_hq_country"),
        contact_email=row.get("contact_email"),
        job_description=row.get("job_description"),
        responsibilities=_ensure_list(row.get("responsibilities")),
        required_qualifications=_ensure_list(row.get("required_qualifications")),
        preferred_qualifications=_ensure_list(row.get("preferred_qualifications")),
        benefits=_ensure_list(row.get("benefits")),
        seniority_level=row.get("seniority_level"),
        remote_policy=row.get("is_remote"),
        remote_label=_remote_label(row.get("is_remote")),
        remote_region=row.get("remote_region"),
        timezone_requirements=row.get("timezone_requirements"),
        work_authorization=row.get("work_authorization"),
        salary=salary_view["salary"],
        salary_range=salary_range,
        salary_currency=salary_view["salary_currency"],
        compensation_type=salary_view["compensation_type"],
        equity_offered=bool(row.get("equity_offered")),
        legitimacy_company_verification=row.get("legitimacy_company_verification"),
        quality_summary=row.get("quality_summary"),
        legitimacy_score=row.get("legitimacy_score"),
        quality_score=row.get("quality_score"),
        apply_url=apply_url,
        apply_url_source=row.get("apply_url_source"),
        posting_url=row.get("posting_url"),
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

# Trim columns to what the frontend needs. job_embedding (vector) and raw_text
# are excluded for payload size.
_SELECT_COLUMNS = ",".join([
    "id",
    "posting_url",
    "apply_url",
    "apply_url_source",
    "source",
    "job_title",
    "company_name",
    "company_hq_country",
    "contact_email",
    "job_description",
    "responsibilities",
    "required_qualifications",
    "preferred_qualifications",
    "benefits",
    "seniority_level",
    "is_remote",
    "remote_region",
    "timezone_requirements",
    "work_authorization",
    "salary_min",
    "salary_max",
    "salary_currency",
    "salary_period",
    "salary_stated",
    "equity_offered",
    "legitimacy_status",
    "legitimacy_score",
    "legitimacy_company_verification",
    "quality_score",
    "quality_summary",
])


def _query_scrapelist(
    *,
    legitimacy_score: int,
    quality_score: int,
    limit: int,
    offset: int,
) -> list[dict]:
    client = get_client()
    end = offset + limit - 1
    response = (
        client.table("scrapelist")
        .select(_SELECT_COLUMNS)
        .gte("legitimacy_score", legitimacy_score)
        .gte("quality_score", quality_score)
        .not_.is_("legitimacy_score", "null")
        .not_.is_("quality_score", "null")
        .order("quality_score", desc=True)
        .order("legitimacy_score", desc=True)
        .range(offset, end)
        .execute()
    )
    return response.data or []


@router.get("/jobs", response_model=JobsResponse)
async def list_jobs(
    legitimacy_score: int = Query(75, ge=0, le=100, description="Minimum legitimacy_score"),
    quality_score: int = Query(80, ge=0, le=100, description="Minimum quality_score"),
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0),
) -> JobsResponse:
    try:
        rows = await asyncio.wait_for(
            asyncio.to_thread(
                _query_scrapelist,
                legitimacy_score=legitimacy_score,
                quality_score=quality_score,
                limit=limit,
                offset=offset,
            ),
            timeout=25.0,
        )
    except asyncio.TimeoutError:
        logger.error(
            "scrapelist query timed out legitimacy>=%s quality>=%s",
            legitimacy_score,
            quality_score,
        )
        raise HTTPException(status_code=504, detail="Job query timed out")
    except Exception as exc:
        logger.exception("scrapelist query failed")
        raise HTTPException(status_code=502, detail=f"Database query failed: {exc}")

    items = [_row_to_item(row) for row in rows]

    return JobsResponse(
        items=items,
        count=len(items),
        limit=limit,
        offset=offset,
        legitimacy_score_min=legitimacy_score,
        quality_score_min=quality_score,
    )
