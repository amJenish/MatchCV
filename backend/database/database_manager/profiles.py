"""Profile reads + idempotency lock helpers."""

from __future__ import annotations

import logging
from typing import Any

from database.database_manager._client import get_client

logger = logging.getLogger(__name__)


class ProfileRepository:
    TABLE = "profiles"

    def _table(self):
        return get_client().table(self.TABLE)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get(self, user_id: str) -> dict | None:
        response = (
            self._table()
            .select("*")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        return rows[0] if rows else None

    def get_signal_profile(self, user_id: str) -> dict | None:
        """Returns the signal_profile JSONB for the user, or None."""
        response = (
            self._table()
            .select("signal_profile")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        if not rows:
            return None
        signal = rows[0].get("signal_profile")
        if not signal or not isinstance(signal, dict):
            return None
        return signal

    def has_signal_profile(self, user_id: str) -> bool:
        return self.get_signal_profile(user_id) is not None

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def update_signal_profile(self, user_id: str, signal_profile: dict) -> None:
        """
        Persist signal_profile JSONB and flip resume_parsed=true. Done as a
        single update so an interrupted upload never leaves resume_parsed=true
        with a null signal_profile.
        """
        if not isinstance(signal_profile, dict):
            raise ValueError("signal_profile must be a dict")
        try:
            self._table().update(
                {
                    "signal_profile": signal_profile,
                    "resume_parsed": True,
                }
            ).eq("id", user_id).execute()
        except Exception:
            logger.exception("update_signal_profile failed for user_id=%s", user_id)
            raise

    # ------------------------------------------------------------------
    # Idempotency lock — prevents duplicate concurrent scraping/discovery
    # for the same user. Implemented as a CAS update; if 0 rows are
    # returned the lock is already held by another in-flight request.
    # ------------------------------------------------------------------

    def is_scraping_in_progress(self, user_id: str) -> bool:
        """Read-only check on the scraping_in_progress flag."""
        try:
            response = (
                self._table()
                .select("scraping_in_progress")
                .eq("id", user_id)
                .limit(1)
                .execute()
            )
        except Exception:
            logger.exception(
                "is_scraping_in_progress read failed for user_id=%s", user_id
            )
            return False
        rows = response.data or []
        if not rows:
            return False
        return bool(rows[0].get("scraping_in_progress"))

    def try_acquire_scraping_lock(self, user_id: str) -> bool:
        try:
            response = (
                self._table()
                .update({"scraping_in_progress": True})
                .eq("id", user_id)
                .eq("scraping_in_progress", False)
                .execute()
            )
        except Exception:
            logger.exception("scraping_in_progress acquire failed for %s", user_id)
            return False
        return bool(response.data)

    def release_scraping_lock(self, user_id: str) -> None:
        try:
            self._table().update({"scraping_in_progress": False}).eq(
                "id", user_id
            ).execute()
        except Exception:
            logger.exception(
                "scraping_in_progress release failed for %s — lock may be stuck",
                user_id,
            )
