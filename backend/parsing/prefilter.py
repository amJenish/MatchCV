from datetime import date, datetime, timedelta

from database.database_manager.scrapelist import ScrapelistRepository
from parsing.content_hash import compute_content_hash
from parsing.job_rules import (
    MAX_POSTING_AGE_DAYS,
    MIN_DESC_LENGTH,
    is_explicitly_inperson,
    is_stub,
)
from parsing.scraper import normalize_url


def prefilter(job: dict, *, seen_hashes: set[str] | None = None) -> bool:
    """
    Cheap local + DB gates run before any LLM call.

    Order matters: cheapest checks first.
    """
    title = job.get("title") or "?"
    repo = ScrapelistRepository()

    posting_url = normalize_url(job.get("url") or "")
    if not posting_url:
        print(f"[Prefilter] REJECT {title}: missing posting URL", flush=True)
        return False

    description = job.get("description") or ""
    if len(description.strip()) < MIN_DESC_LENGTH:
        print(
            f"[Prefilter] REJECT {title}: description too short "
            f"({len(description.strip())} < {MIN_DESC_LENGTH})",
            flush=True,
        )
        return False

    if is_stub(title, description):
        print(f"[Prefilter] REJECT {title}: aggregator/LinkedIn stub", flush=True)
        return False

    if is_explicitly_inperson(
        title,
        description,
        job.get("location") or "",
    ):
        print(f"[Prefilter] REJECT {title}: explicitly in-person", flush=True)
        return False

    if _is_too_old(job.get("date_posted")):
        print(
            f"[Prefilter] REJECT {title}: posting too old or missing date "
            f"({job.get('date_posted')!r}, max {MAX_POSTING_AGE_DAYS} days)",
            flush=True,
        )
        return False

    content_hash = compute_content_hash(job)
    if seen_hashes is not None:
        if content_hash in seen_hashes:
            print(f"[Prefilter] REJECT {title}: content already seen this run", flush=True)
            return False
    if repo.content_hash_exists(content_hash):
        print(f"[Prefilter] REJECT {title}: content already in scrapelist", flush=True)
        return False
    if repo.posting_url_exists(posting_url):
        print(f"[Prefilter] REJECT {title}: posting_url already in scrapelist", flush=True)
        return False

    if seen_hashes is not None:
        seen_hashes.add(content_hash)
    job["_content_hash"] = content_hash
    job["url"] = posting_url

    print(f"[Prefilter] PASS {title}", flush=True)
    return True


def _is_too_old(date_posted: str | None) -> bool:
    if not date_posted:
        return True
    try:
        posted = datetime.fromisoformat(str(date_posted)[:10]).date()
    except ValueError:
        return True
    return (date.today() - posted) > timedelta(days=MAX_POSTING_AGE_DAYS)
