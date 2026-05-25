"""user_job_interactions repository.

The (user_id, job_id) unique constraint guarantees we never resurface a
job to the same user — regardless of status. shown_order is monotonic
per-user; assignment is serialized via the profiles scraping lock.

Joins to scrapelist are done as an explicit two-step fetch rather than
PostgREST's resource embedding so we don't depend on a specific FK
constraint name in the live database.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from database.database_manager._client import get_client

logger = logging.getLogger(__name__)

CAP = 20


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_INTERACTION_COLUMNS = (
    "id, status, shown_order, fit_reason, fit_score, "
    "legitimacy_score, quality_score, composite_score, "
    "job_id, acted_at, created_at"
)


def _attach_jobs(rows: list[dict]) -> list[dict]:
    """Take user_job_interactions rows and inline the matching scrapelist
    row under the `job` key. One extra round-trip; no FK-name dependency."""
    if not rows:
        return []
    job_ids = [r["job_id"] for r in rows if r.get("job_id")]
    if not job_ids:
        return [{**r, "job": None} for r in rows]

    client = get_client()
    response = (
        client.table("scrapelist")
        .select("*")
        .in_("id", job_ids)
        .execute()
    )
    jobs_by_id = {j["id"]: j for j in (response.data or [])}

    return [{**r, "job": jobs_by_id.get(r.get("job_id"))} for r in rows]


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

class UserJobInteractionsRepository:
    TABLE = "user_job_interactions"

    def _table(self):
        return get_client().table(self.TABLE)

    # ------------------------------------------------------------------
    # Counts / queries
    # ------------------------------------------------------------------

    def count_shown(self, user_id: str) -> int:
        response = (
            self._table()
            .select("id", count="exact")
            .eq("user_id", user_id)
            .eq("status", "shown")
            .execute()
        )
        return int(getattr(response, "count", 0) or 0)

    def max_shown_order(self, user_id: str) -> int:
        response = (
            self._table()
            .select("shown_order")
            .eq("user_id", user_id)
            .order("shown_order", desc=True)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        if not rows:
            return 0
        return int(rows[0].get("shown_order") or 0)

    def has_interaction(self, user_id: str, job_id: str) -> bool:
        """Return True if this user already has any-status row for this job."""
        if not job_id:
            return False
        response = (
            self._table()
            .select("id")
            .eq("user_id", user_id)
            .eq("job_id", job_id)
            .limit(1)
            .execute()
        )
        return bool(response.data)

    def list_shown_with_jobs(self, user_id: str) -> list[dict]:
        rows = (
            self._table()
            .select(_INTERACTION_COLUMNS)
            .eq("user_id", user_id)
            .eq("status", "shown")
            .order("shown_order", desc=False)
            .execute()
            .data or []
        )
        return _attach_jobs(rows)

    def list_applied_with_jobs(self, user_id: str) -> list[dict]:
        rows = (
            self._table()
            .select(_INTERACTION_COLUMNS)
            .eq("user_id", user_id)
            .eq("status", "applied")
            .order("acted_at", desc=True)
            .execute()
            .data or []
        )
        return _attach_jobs(rows)

    # ------------------------------------------------------------------
    # Inserts
    # ------------------------------------------------------------------

    def insert_shown_one(
        self,
        user_id: str,
        payload: dict,
        shown_order: int,
    ) -> dict | None:
        """
        Insert a single 'shown' row. Used by the streaming discovery path
        so each newly-found job appears in the user's queue immediately.

        On (user_id, job_id) collision the unique constraint silently
        ignores the duplicate and returns None.
        """
        if not payload.get("job_id"):
            return None
        row = {
            "user_id":          user_id,
            "job_id":           payload["job_id"],
            "status":           "shown",
            "shown_order":      shown_order,
            "fit_reason":       payload.get("fit_reason"),
            "fit_score":        payload.get("fit_score"),
            "legitimacy_score": payload.get("legitimacy_score"),
            "quality_score":    payload.get("quality_score"),
            "composite_score":  payload.get("composite_score"),
        }
        try:
            response = (
                self._table()
                .upsert(
                    [row],
                    on_conflict="user_id,job_id",
                    ignore_duplicates=True,
                )
                .execute()
            )
        except Exception:
            logger.exception(
                "insert_shown_one failed user_id=%s job_id=%s",
                user_id,
                payload.get("job_id"),
            )
            raise

        rows = response.data or []
        return rows[0] if rows else None

    def bulk_insert_shown(
        self,
        user_id: str,
        items: list[dict],
        *,
        starting_order: int,
    ) -> list[dict]:
        """Legacy batch insert used by older paths. Prefer insert_shown_one."""
        if not items:
            return []

        rows: list[dict] = []
        for offset, item in enumerate(items, start=1):
            rows.append(
                {
                    "user_id":          user_id,
                    "job_id":           item["job_id"],
                    "status":           "shown",
                    "shown_order":      starting_order + offset,
                    "fit_reason":       item.get("fit_reason"),
                    "fit_score":        item.get("fit_score"),
                    "legitimacy_score": item.get("legitimacy_score"),
                    "quality_score":    item.get("quality_score"),
                    "composite_score":  item.get("composite_score"),
                }
            )

        try:
            response = (
                self._table()
                .upsert(rows, on_conflict="user_id,job_id", ignore_duplicates=True)
                .execute()
            )
        except Exception:
            logger.exception("bulk_insert_shown failed for user_id=%s", user_id)
            raise

        return response.data or []

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def update_status(
        self,
        interaction_id: str,
        user_id: str,
        status: str,
    ) -> dict | None:
        if status not in {"applied", "not_interested", "saved"}:
            raise ValueError(f"Refusing to set unsupported status: {status!r}")

        now_iso = datetime.now(timezone.utc).isoformat()
        response = (
            self._table()
            .update(
                {
                    "status": status,
                    "acted_at": now_iso,
                    "updated_at": now_iso,
                }
            )
            .eq("id", interaction_id)
            .eq("user_id", user_id)
            .execute()
        )
        rows = response.data or []
        return rows[0] if rows else None
