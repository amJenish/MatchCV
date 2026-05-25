"""Supabase Google OAuth — /auth/login, /auth/callback, /auth/me, refresh, logout."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from supabase_auth.errors import AuthApiError

from database.database_manager._client import get_client as get_db_client
from database.database_manager.auth_client import get_auth_client

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")

router = APIRouter(prefix="/auth", tags=["auth"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RefreshBody(BaseModel):
    refresh_token: str


class ErrorResponse(BaseModel):
    error: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _redirect_uri() -> str:
    uri = os.getenv("SUPABASE_REDIRECT_URI", "http://127.0.0.1:8000/auth/callback")
    return uri.strip()


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        return None
    token = authorization[len(prefix) :].strip()
    return token or None


def _metadata_names(user: Any) -> tuple[str | None, str | None]:
    meta = getattr(user, "user_metadata", None) or {}
    if not isinstance(meta, dict):
        meta = {}
    full_name = meta.get("full_name") or meta.get("name")
    last_name = meta.get("last_name")
    if not last_name and full_name and isinstance(full_name, str):
        parts = full_name.strip().split()
        if len(parts) > 1:
            last_name = parts[-1]
    return (
        str(full_name).strip() if full_name else None,
        str(last_name).strip() if last_name else None,
    )


def _user_payload(user: Any, *, full_name: str | None = None, last_name: str | None = None) -> dict:
    fn, ln = _metadata_names(user)
    return {
        "id": str(user.id),
        "email": getattr(user, "email", None),
        "full_name": full_name if full_name is not None else fn,
        "last_name": last_name if last_name is not None else ln,
    }


def _ensure_profile(user: Any) -> None:
    """Insert profiles row if missing. id must equal auth.users.id."""
    user_id = str(user.id)
    db = get_db_client()
    existing = (
        db.table("profiles")
        .select("id")
        .eq("id", user_id)
        .limit(1)
        .execute()
    )
    if existing.data:
        return

    full_name, last_name = _metadata_names(user)
    row = {
        "id": user_id,
        "full_name": full_name,
        "last_name": last_name,
        "email": getattr(user, "email", None),
    }
    db.table("profiles").insert(row).execute()
    logger.info("Created profile for user_id=%s", user_id)


def _validate_token(access_token: str) -> Any:
    """Return Supabase user from access token; raise HTTPException on failure."""
    try:
        client = get_auth_client()
        response = client.auth.get_user(access_token)
    except Exception as exc:
        logger.exception("get_user failed during token validation")
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc

    if not response or not getattr(response, "user", None):
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return response.user


def require_user(authorization: str | None = Header(None)) -> Any:
    """FastAPI dependency: valid Bearer access token required."""
    token = _bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    return _validate_token(token)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/login")
def login() -> JSONResponse:
    try:
        client = get_auth_client()
        oauth = client.auth.sign_in_with_oauth(
            {
                "provider": "google",
                "options": {"redirect_to": _redirect_uri()},
            }
        )
        if not oauth or not getattr(oauth, "url", None):
            logger.error("sign_in_with_oauth returned no URL")
            return _error(502, "Could not start Google sign-in")
        return JSONResponse(content={"url": oauth.url})
    except AuthApiError as exc:
        logger.error("Supabase OAuth error code=%s status=%s", exc.code, getattr(exc, "status", None))
        msg = getattr(exc, "message", None) or str(exc) or "Could not start Google sign-in"
        if "not enabled" in msg.lower() or exc.code == "validation_failed":
            msg = (
                "Google sign-in is not enabled in your Supabase project. "
                "Open Supabase Dashboard → Authentication → Providers → Google, "
                "turn it on, and add your Google OAuth Client ID and Secret."
            )
        return _error(400, msg)
    except Exception:
        logger.exception("OAuth login initiation failed")
        return _error(502, "Could not start Google sign-in")


@router.get("/callback", response_model=None)
def callback(
    request: Request,
    code: str | None = Query(None),
    state: str | None = Query(None),
):
    if not code:
        logger.warning("OAuth callback missing code param state=%s", state)
        return _error(400, "Missing authorization code")

    redirect_to = _redirect_uri()
    try:
        client = get_auth_client()
        auth_response = client.auth.exchange_code_for_session(
            {
                "auth_code": code,
                "redirect_to": redirect_to,
            }
        )
    except Exception as exc:
        logger.exception(
            "exchange_code_for_session failed redirect_to=%s state=%s",
            redirect_to,
            state,
        )
        return _error(401, "Could not complete sign-in")

    session = getattr(auth_response, "session", None)
    user = getattr(auth_response, "user", None) or (
        getattr(session, "user", None) if session else None
    )
    if not session or not user:
        logger.error("OAuth callback returned no session or user")
        return _error(401, "Could not complete sign-in")

    try:
        _ensure_profile(user)
    except Exception:
        logger.exception("Profile insert failed for user_id=%s", getattr(user, "id", None))
        return _error(502, "Could not create user profile")

    access_token = session.access_token
    refresh_token = session.refresh_token
    payload = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "user": _user_payload(user),
    }

    # Browser OAuth redirect: send tokens to profile page (stripped client-side).
    accept = request.headers.get("accept", "")
    if "text/html" in accept and "application/json" not in accept:
        params = urlencode(
            {
                "access_token": access_token,
                "refresh_token": refresh_token,
            }
        )
        return RedirectResponse(url=f"/?{params}", status_code=302)

    return JSONResponse(content=payload)


@router.get("/me")
def me(authorization: str | None = Header(None)) -> JSONResponse:
    token = _bearer_token(authorization)
    if not token:
        return _error(401, "Invalid or expired token")

    try:
        user = _validate_token(token)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else "Invalid or expired token"
        return _error(401, detail)

    try:
        db = get_db_client()
        result = (
            db.table("profiles")
            .select(
                "id, email, full_name, last_name, resume_parsed, "
                "signal_profile, created_at"
            )
            .eq("id", str(user.id))
            .limit(1)
            .execute()
        )
    except Exception:
        logger.exception("profiles query failed for user_id=%s", user.id)
        return _error(502, "Could not load profile")

    rows = result.data or []
    if not rows:
        return _error(404, "Profile not found")

    row = rows[0]
    signal = row.get("signal_profile")
    has_signal = isinstance(signal, dict) and bool(signal)
    return JSONResponse(
        content={
            "id": row.get("id"),
            "email": row.get("email"),
            "full_name": row.get("full_name"),
            "last_name": row.get("last_name"),
            "resume_parsed": bool(row.get("resume_parsed")),
            "has_signal_profile": has_signal,
            "signal_profile": signal if has_signal else None,
            "created_at": row.get("created_at"),
        }
    )


@router.post("/refresh")
def refresh(body: RefreshBody) -> JSONResponse:
    if not body.refresh_token.strip():
        return _error(400, "refresh_token is required")

    try:
        client = get_auth_client()
        auth_response = client.auth.refresh_session(body.refresh_token.strip())
    except Exception:
        logger.exception("refresh_session failed")
        return _error(401, "Could not refresh session")

    session = getattr(auth_response, "session", None)
    if not session:
        return _error(401, "Could not refresh session")

    return JSONResponse(
        content={
            "access_token": session.access_token,
            "refresh_token": session.refresh_token,
        }
    )


@router.post("/logout")
def logout(authorization: str | None = Header(None)) -> JSONResponse:
    token = _bearer_token(authorization)
    if not token:
        return _error(401, "Invalid or expired token")

    try:
        user = _validate_token(token)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else "Invalid or expired token"
        return _error(401, detail)

    try:
        client = get_auth_client()
        # Bearer-only logout: admin.sign_out revokes refresh tokens for this JWT.
        client.auth.admin.sign_out(token, "global")
    except Exception:
        logger.exception("sign_out failed for user_id=%s", getattr(user, "id", None))
        return _error(502, "Could not sign out")

    return JSONResponse(content={"message": "Logged out successfully"})
