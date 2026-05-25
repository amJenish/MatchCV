"""Repository for the four resume-section tables.

Every upload performs a delete-then-insert per user so we never leave
stale rows from earlier resume revisions.

Live schema (per the user's tables):
  work_experience: work_title text NOT NULL, company text NOT NULL,
                   work_start date NOT NULL, work_end date NULL,
                   is_current boolean default false, work_info text[]
  projects:        project_title text, project_info text[], tech_stack text[]
  education:       institution text, degree text, field_of_study text,
                   graduation_year integer
  skills:          skills text[] NOT NULL default '{}'
"""

from __future__ import annotations

import logging
import re
from datetime import date

from database.database_manager._client import get_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------

_MONTHS: dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4,
    "june": 6, "july": 7, "august": 8, "september": 9,
    "october": 10, "november": 11, "december": 12,
}

# Words that mean "still ongoing" — we map them to (None, is_current=True).
_PRESENT_TOKENS = {"present", "current", "now", "ongoing", "today", "currently"}


def _norm_text_array(value) -> list[str]:
    """Coerce model output into a clean list[str] for Postgres text[] columns."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(x).strip() for x in value if x is not None and str(x).strip()]
    return []


def _to_iso_date(value) -> str | None:
    """
    Best-effort coercion of arbitrary resume date strings to YYYY-MM-DD.
    Returns None on failure or for "Present"-style values.
    Accepts: '2022-03-15', '2022-03', '2022', 'Jan 2022', 'January 2022'.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.lower() in _PRESENT_TOKENS:
        return None

    # YYYY-MM-DD or YYYY-MM (Claude is told to emit this)
    m = re.match(r"^(\d{4})-(\d{1,2})(?:-(\d{1,2}))?$", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3) or 1)).isoformat()
        except ValueError:
            return None

    # YYYY only
    m = re.match(r"^(\d{4})$", s)
    if m:
        try:
            return date(int(m.group(1)), 1, 1).isoformat()
        except ValueError:
            return None

    # "Mon YYYY" / "Month YYYY"
    m = re.match(r"^([A-Za-z]+)\s+(\d{4})$", s)
    if m:
        month_name = m.group(1).lower()
        if month_name in _MONTHS:
            try:
                return date(int(m.group(2)), _MONTHS[month_name], 1).isoformat()
            except ValueError:
                return None

    # "YYYY Mon" (rare resume style)
    m = re.match(r"^(\d{4})\s+([A-Za-z]+)$", s)
    if m:
        month_name = m.group(2).lower()
        if month_name in _MONTHS:
            try:
                return date(int(m.group(1)), _MONTHS[month_name], 1).isoformat()
            except ValueError:
                return None

    return None


def _looks_present(value) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in _PRESENT_TOKENS


def _to_year_int(value) -> int | None:
    """Coerce to a plausible 4-digit year integer, else None."""
    if value is None:
        return None
    if isinstance(value, int):
        return value if 1900 <= value <= 2100 else None
    s = str(value).strip()
    m = re.match(r"^(\d{4})$", s)
    if m:
        try:
            year = int(m.group(1))
        except ValueError:
            return None
        return year if 1900 <= year <= 2100 else None
    return None


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

class ResumeSectionsRepository:

    def _client(self):
        return get_client()

    # ------------------------------------------------------------------
    # Replace = delete then insert (atomic per-table from the user's POV
    # because the upload is serialized by the resume endpoint).
    # ------------------------------------------------------------------

    def replace_work_experience(self, user_id: str, items: list[dict]) -> int:
        client = self._client()
        client.table("work_experience").delete().eq("user_id", user_id).execute()

        rows: list[dict] = []
        for item in items or []:
            work_title = (item.get("work_title") or "").strip()
            company    = (item.get("company") or "").strip()
            work_start_iso = _to_iso_date(item.get("work_start"))
            work_end_iso   = _to_iso_date(item.get("work_end"))

            # Live schema is NOT NULL on work_title, company, work_start.
            # Skip rather than 409 on the first incomplete row.
            if not work_title or not company or not work_start_iso:
                logger.info(
                    "replace_work_experience: skipping incomplete row "
                    "(title=%r company=%r start=%r) for user_id=%s",
                    work_title or None,
                    company or None,
                    item.get("work_start"),
                    user_id,
                )
                continue

            # is_current: trust Claude's bool; otherwise derive from
            # 'Present'-style strings or a missing end date.
            is_current_raw = item.get("is_current")
            if isinstance(is_current_raw, bool):
                is_current = is_current_raw
            else:
                is_current = (
                    _looks_present(item.get("work_end"))
                    or work_end_iso is None
                )

            rows.append(
                {
                    "user_id":    user_id,
                    "work_title": work_title,
                    "company":    company,
                    "work_start": work_start_iso,
                    "work_end":   work_end_iso,
                    "is_current": is_current,
                    "work_info":  _norm_text_array(item.get("work_info")),
                }
            )

        if not rows:
            return 0
        response = client.table("work_experience").insert(rows).execute()
        return len(response.data or [])

    def replace_projects(self, user_id: str, items: list[dict]) -> int:
        client = self._client()
        client.table("projects").delete().eq("user_id", user_id).execute()

        rows = [
            {
                "user_id":       user_id,
                # Column is `project_title` in the live schema (matches the
                # rest of the codebase: prefilter, fit checker, adapter).
                "project_title": (item.get("project_title") or item.get("project_name") or None),
                "project_info":  _norm_text_array(item.get("project_info")),
                "tech_stack":    _norm_text_array(item.get("tech_stack")),
            }
            for item in (items or [])
            if (
                item.get("project_title")
                or item.get("project_name")
                or item.get("project_info")
                or item.get("tech_stack")
            )
        ]
        if not rows:
            return 0
        response = client.table("projects").insert(rows).execute()
        return len(response.data or [])

    def replace_education(self, user_id: str, items: list[dict]) -> int:
        client = self._client()
        client.table("education").delete().eq("user_id", user_id).execute()

        rows = [
            {
                "user_id":         user_id,
                "institution":     (item.get("institution") or None),
                "degree":          (item.get("degree") or None),
                "field_of_study":  (item.get("field_of_study") or None),
                # Live column is integer.
                "graduation_year": _to_year_int(item.get("graduation_year")),
            }
            for item in (items or [])
            if (item.get("institution") or item.get("degree") or item.get("field_of_study"))
        ]
        if not rows:
            return 0
        response = client.table("education").insert(rows).execute()
        return len(response.data or [])

    def replace_skills(self, user_id: str, skills: list[str]) -> int:
        """skills is a flat list. We store one row per user with the full array."""
        client = self._client()
        client.table("skills").delete().eq("user_id", user_id).execute()

        cleaned = _norm_text_array(skills)
        if not cleaned:
            return 0
        response = client.table("skills").insert(
            {"user_id": user_id, "skills": cleaned}
        ).execute()
        return len(response.data or [])

    # ------------------------------------------------------------------
    # Read all sections for a user — used by SignalExtractor.
    # ------------------------------------------------------------------

    def get_all_for_user(self, user_id: str) -> dict:
        client = self._client()
        we = (
            client.table("work_experience")
            .select("work_title, company, work_start, work_end, is_current, work_info")
            .eq("user_id", user_id)
            .execute()
        )
        proj = (
            client.table("projects")
            .select("project_title, project_info, tech_stack")
            .eq("user_id", user_id)
            .execute()
        )
        edu = (
            client.table("education")
            .select("institution, degree, field_of_study, graduation_year")
            .eq("user_id", user_id)
            .execute()
        )
        sk = (
            client.table("skills")
            .select("skills")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )

        skills_rows = sk.data or []
        flat_skills = skills_rows[0]["skills"] if skills_rows else []

        return {
            "work_experience": we.data or [],
            "projects":        proj.data or [],
            "education":       edu.data or [],
            "skills":          flat_skills,
        }
