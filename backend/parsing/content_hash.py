import hashlib


def compute_content_hash(job: dict) -> str:
    """
    SHA-256 over the normalized title + first 2 KB of description, so
    cross-board duplicates of the same posting collapse to one row.
    """
    title = (job.get("title") or "").strip().lower()
    desc = (job.get("description") or "").strip().lower()[:2000]
    payload = f"{title}|{desc}".encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()
