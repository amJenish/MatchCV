"""
Cheap fit prefilter: keyword overlap + local sentence-transformer cosine
across the profile's work / project / education dimensions vs each job.
No LLM calls, no paid embedding API — runs `all-MiniLM-L6-v2` on CPU.

Used by JobDiscoveryService to narrow a large candidate pool down to the top
N before running the expensive Anthropic-based JobFitChecker rubric.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
from collections import Counter

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_BATCH = 32
MAX_CHARS_PER_TEXT = 4_000  # MiniLM truncates at 256 tokens; leave headroom.

WORK_WEIGHT = 0.50
PROJECT_WEIGHT = 0.35
EDUCATION_WEIGHT = 0.15

KEYWORD_WEIGHT = 0.50
EMBEDDING_WEIGHT = 0.50


# ---------------------------------------------------------------------------
# SentenceTransformer model (lazy singleton)
# ---------------------------------------------------------------------------

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load the embedding model on first use (~80 MB download once)."""
    global _model
    if _model is None:
        logger.info("Loading sentence-transformer %s...", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
        dim = _model.get_sentence_embedding_dimension()
        if dim != EMBEDDING_DIM:
            raise RuntimeError(
                f"Embedding model returned {dim}-dim vectors but "
                f"EMBEDDING_DIM={EMBEDDING_DIM}. Update the constant and "
                "the scrapelist.job_embedding column."
            )
    return _model


# ---------------------------------------------------------------------------
# Tokenization & similarity
# ---------------------------------------------------------------------------

_TOKEN_RX = re.compile(r"\b[a-z][a-z0-9+#.]*\b")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RX.findall((text or "").lower())


def _overlap_score(profile_tokens: list[str], job_tokens: list[str]) -> float:
    """Overlap coefficient using min(profile_len, job_len) as denominator."""
    if not profile_tokens or not job_tokens:
        return 0.0
    pcounts = Counter(profile_tokens)
    jcounts = Counter(job_tokens)
    shared = sum((pcounts & jcounts).values())
    denom = min(sum(pcounts.values()), sum(jcounts.values()))
    return shared / denom if denom else 0.0


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    ma = math.sqrt(sum(x * x for x in a))
    mb = math.sqrt(sum(x * x for x in b))
    if ma == 0 or mb == 0:
        return 0.0
    return dot / (ma * mb)


# ---------------------------------------------------------------------------
# Profile / job text builders
# ---------------------------------------------------------------------------

def _work_text(profile: dict) -> str:
    parts: list[str] = []
    work = profile.get("work_experience") or []
    for exp in sorted(work, key=lambda x: x.get("work_start") or "", reverse=True):
        title = exp.get("work_title") or ""
        company = exp.get("company") or ""
        info = " ".join(exp.get("work_info") or [])
        chunk = " ".join(p for p in (title, company, info) if p)
        if chunk:
            parts.append(chunk)
    return " ".join(parts)


def _project_text(profile: dict) -> str:
    parts: list[str] = []
    for proj in profile.get("projects") or []:
        title = proj.get("project_title") or ""
        info = " ".join(proj.get("project_info") or [])
        stack = " ".join(proj.get("tech_stack") or [])
        chunk = " ".join(p for p in (title, info, stack) if p)
        if chunk:
            parts.append(chunk)
    return " ".join(parts)


def _education_text(profile: dict) -> str:
    parts: list[str] = []
    for edu in profile.get("education") or []:
        cred = (
            edu.get("credential_name")
            or edu.get("degree")
            or edu.get("credential_type")
            or ""
        )
        inst = edu.get("institution") or ""
        field = edu.get("field_of_study") or ""
        chunk = " ".join(p for p in (cred, inst, field) if p)
        if chunk:
            parts.append(chunk)
    return " ".join(parts)


def _job_text(job: dict) -> str:
    parts = [
        job.get("job_title") or "",
        job.get("job_description") or "",
        " ".join(job.get("responsibilities") or []),
        " ".join(job.get("required_qualifications") or []),
        " ".join(job.get("preferred_qualifications") or []),
    ]
    return " ".join(p for p in parts if p)


def _truncate(text: str) -> str:
    text = text or ""
    return text[:MAX_CHARS_PER_TEXT]


# ---------------------------------------------------------------------------
# Embeddings (with profile-level cache)
# ---------------------------------------------------------------------------

_PROFILE_EMB_CACHE: dict[str, dict[str, list[float]]] = {}


def _profile_cache_key(work: str, proj: str, edu: str) -> str:
    h = hashlib.sha256()
    h.update(work.encode("utf-8", errors="ignore"))
    h.update(b"||")
    h.update(proj.encode("utf-8", errors="ignore"))
    h.update(b"||")
    h.update(edu.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _encode_sync(texts: list[str]) -> list[list[float]]:
    """Run encode in the calling thread. Used inside asyncio.to_thread."""
    model = _get_model()
    arrs = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    return [[float(x) for x in vec] for vec in arrs]


async def _embed(texts: list[str]) -> list[list[float]]:
    """Embed texts on a worker thread to keep the event loop responsive."""
    if not texts:
        return []
    payload = [_truncate(t) or " " for t in texts]
    return await asyncio.to_thread(_encode_sync, payload)


async def embed_profile(profile: dict) -> tuple[dict[str, list[float]], dict[str, str]]:
    """
    Returns ({"work": vec, "proj": vec, "edu": vec}, {"work": text, ...}).
    Cached in-process by hash of the canonical profile text — repeated calls
    for the same profile cost zero API.
    """
    work = _work_text(profile)
    proj = _project_text(profile)
    edu = _education_text(profile)

    key = _profile_cache_key(work, proj, edu)
    cached = _PROFILE_EMB_CACHE.get(key)
    texts = {"work": work, "proj": proj, "edu": edu}
    if cached is not None:
        logger.debug("embed_profile cache hit (%s)", key[:8])
        return cached, texts

    embeds = await _embed([work or " ", proj or " ", edu or " "])
    embeddings = {"work": embeds[0], "proj": embeds[1], "edu": embeds[2]}
    _PROFILE_EMB_CACHE[key] = embeddings
    return embeddings, texts


async def ensure_job_embeddings(
    jobs: list[dict],
) -> list[tuple[str, list[float]]]:
    """
    Fill `job_embedding` on every job dict that doesn't already have one.
    Returns the list of (id, embedding) pairs the caller should persist.
    """
    needs = [j for j in jobs if not j.get("job_embedding")]
    if not needs:
        return []

    texts = [_job_text(j) for j in needs]
    embeds = await _embed(texts)

    persisted: list[tuple[str, list[float]]] = []
    for job, emb in zip(needs, embeds):
        job["job_embedding"] = emb
        if job.get("id"):
            persisted.append((job["id"], emb))
    return persisted


# ---------------------------------------------------------------------------
# Prefilter
# ---------------------------------------------------------------------------

def _score_one(
    job: dict,
    profile_tokens: dict[str, list[str]],
    profile_emb: dict[str, list[float]],
) -> float:
    job_tokens = _tokenize(_job_text(job))
    job_emb = job.get("job_embedding") or []

    dim_scores = {}
    for dim in ("work", "proj", "edu"):
        kw = _overlap_score(profile_tokens[dim], job_tokens)
        cs = _cosine(profile_emb[dim], job_emb)
        dim_scores[dim] = KEYWORD_WEIGHT * kw + EMBEDDING_WEIGHT * cs

    return (
        WORK_WEIGHT * dim_scores["work"]
        + PROJECT_WEIGHT * dim_scores["proj"]
        + EDUCATION_WEIGHT * dim_scores["edu"]
    )


async def prefilter_top_n(
    profile: dict,
    jobs: list[dict],
    n: int = 40,
) -> list[tuple[dict, float]]:
    """
    Score each job against the profile and return the top N (job, score)
    pairs sorted best -> worst.

    Caller is responsible for calling `ensure_job_embeddings` first if it
    wants persistence; this function will accept jobs whose `job_embedding`
    has already been populated and skip the embedding step entirely.
    """
    if not jobs:
        return []

    profile_emb, profile_text = await embed_profile(profile)

    # Embedding fallback if any job is missing its cached vector.
    if any(not j.get("job_embedding") for j in jobs):
        await ensure_job_embeddings(jobs)

    profile_tokens = {
        "work": _tokenize(profile_text["work"]),
        "proj": _tokenize(profile_text["proj"]),
        "edu": _tokenize(profile_text["edu"]),
    }

    scored = [
        (job, _score_one(job, profile_tokens, profile_emb))
        for job in jobs
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:n]
