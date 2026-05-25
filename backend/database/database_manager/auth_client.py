"""Supabase client for Auth API (OAuth, sessions)."""

import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from supabase import Client, create_client

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env")


def _resolve_supabase_key() -> tuple[str | None, str]:
    """Return (api_key, source_label). Prefers anon/public key for OAuth."""
    if key := os.getenv("SUPABASE_KEY"):
        return key, "SUPABASE_KEY"
    if key := os.getenv("SUPABASE_ANON_KEY"):
        return key, "SUPABASE_ANON_KEY"
    if key := os.getenv("SUPABASE_SECRET_KEY"):
        logger.warning(
            "SUPABASE_KEY not set; using SUPABASE_SECRET_KEY for auth. "
            "Add SUPABASE_KEY (anon/public) from Supabase Dashboard → API for OAuth."
        )
        return key, "SUPABASE_SECRET_KEY"
    return None, "none"


@lru_cache
def get_auth_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key, source = _resolve_supabase_key()
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and a Supabase API key must be set in backend/.env. "
            "Set SUPABASE_KEY (anon/public, recommended for OAuth) or SUPABASE_SECRET_KEY."
        )
    logger.debug("Auth client using key from %s", source)
    return create_client(url, key)
