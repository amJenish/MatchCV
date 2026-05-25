"""FastAPI entrypoint. Run with `uvicorn api.main:app --reload` from backend/."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.auth import router as auth_router
from api.jobs import router as jobs_router
from api.queue import router as queue_router
from api.resume import router as resume_router

logging.basicConfig(level=logging.INFO)

# Project root — frontend/ lives next to backend/. Folder name on disk
# (currently "Career Guardian") is left intentionally; only user-visible
# branding is renamed to MatchCV.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND_DIR = _PROJECT_ROOT / "frontend"

app = FastAPI(title="MatchCV API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(resume_router)
# queue_router uses the same /api/jobs prefix with sub-paths /queue, /interact,
# /find-more, /applied — included before the catch-all jobs_router so the
# specific routes win the match order.
app.include_router(queue_router)
app.include_router(jobs_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _frontend_page(name: str):
    path = _FRONTEND_DIR / name
    if not path.is_file():
        logging.warning("Missing frontend page: %s", path)
        return None
    return FileResponse(path)


@app.get("/")
def serve_index():
    """HTML shell; auth enforced client-side and on /api/*."""
    page = _frontend_page("index.html")
    if page is None:
        return {"error": "frontend/index.html not found"}
    return page


@app.get("/auth.js")
def serve_auth_js():
    page = _frontend_page("auth.js")
    if page is None:
        return {"error": "frontend/auth.js not found"}
    return page


@app.get("/login.html")
def serve_login():
    page = _frontend_page("login.html")
    if page is None:
        return {"error": "frontend/login.html not found"}
    return page


@app.get("/profile.html")
def serve_profile():
    page = _frontend_page("profile.html")
    if page is None:
        return {"error": "frontend/profile.html not found"}
    return page
