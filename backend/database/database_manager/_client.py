import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from supabase import Client, create_client

_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env")


@lru_cache
def get_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SECRET_KEY")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SECRET_KEY must be set in backend/.env"
        )
    return create_client(url, key)
