import asyncio
import json
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from itertools import zip_longest
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup

from parsing.apply_urls import (
    dedupe_jobs_merge_apply,
    merge_apply_url,
    parse_api_apply_url,
)
from parsing.constants import QUEUE_END

logger = logging.getLogger(__name__)

REMOTEOK_URL = "https://remoteok.com/api"
REMOTIVE_URL = "https://remotive.com/api/remote-jobs"
WEWORKREMOTELY_RSS = "https://weworkremotely.com/categories/remote-programming-jobs.rss"
HIMALAYAS_URL = "https://himalayas.app/jobs/api"
ARBEITNOW_URL = "https://www.arbeitnow.com/api/job-board-api"

# RemoteOK appends an anti-spam suffix asking applicants to mention a keyword
# and tag ROjox. Strip it so it never reaches Extraction or Quality.
ROJOX_PATTERN = re.compile(
    r"(?:<br\s*/?>\s*)*\s*Please mention the word\s+\*?\*?\w+\*?\*?\s+and tag.*",
    re.IGNORECASE | re.DOTALL,
)

# Tracking query parameters dropped during URL canonicalization.
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "ref", "ref_src", "fbclid", "gclid", "mc_cid", "mc_eid",
}

# Generic terms used when the caller didn't provide any signal-derived
# keywords (broad-firehose mode). Arbeitnow's API requires a search term to
# return results, so we feed it these instead of an empty list.
DEFAULT_SECONDARY_TERMS = ["developer", "engineer", "software"]

# Pre-pipeline relaxation thresholds. The scraper builds a candidate pool
# before emitting anything; if the pool is smaller than MIN_CANDIDATES we
# progressively relax (top-K keywords, then unfiltered) without paying any
# downstream LLM cost. Tuned conservatively so the streaming consumer has
# enough material to find 5 fit-passing jobs after the verify pipeline.
MIN_CANDIDATES = 30
TOP_K_KEYWORDS = 3

# Remotive detail-page apply extraction (best-effort; breaks if HTML changes).
REMOTIVE_ENRICH_CONCURRENCY = 3
REMOTIVE_ENRICH_DELAY_SEC = 0.35

_REMOTEOK_SAMPLE_LOGGED = False

# Cache for compiled word-boundary regexes used by the keyword post-filter.
_WORD_BOUNDARY_CACHE: dict[str, re.Pattern[str]] = {}


def normalize_url(url: str) -> str:
    """
    Canonicalize a posting URL: lowercase scheme/host, strip trailing slash,
    drop tracking params. Returns "" if the input is unparseable.
    """
    if not url:
        return ""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return ""
    if not parts.scheme or not parts.netloc:
        return ""

    if parts.query:
        kept = [
            kv for kv in parts.query.split("&")
            if kv and kv.split("=", 1)[0].lower() not in TRACKING_PARAMS
        ]
        query = "&".join(kept)
    else:
        query = ""

    path = parts.path.rstrip("/") or ""

    return urlunsplit((
        parts.scheme.lower(),
        parts.netloc.lower(),
        path,
        query,
        "",
    ))


def _empty_job() -> dict:
    return {
        "title": None,
        "company": None,
        "description": None,
        "url": None,
        "apply_url": None,
        "apply_url_source": None,
        "location": None,
        "tags": [],
        "salary": None,
        "date_posted": None,
        "source": None,
    }


def _set_posting_urls(
    job: dict,
    *,
    posting_url: str,
    apply_url: str | None = None,
    apply_url_source: str | None = None,
) -> None:
    """url is the board listing (dedup key); apply_url is where candidates apply."""
    job["url"] = posting_url
    if apply_url:
        job["apply_url"] = apply_url
        job["apply_url_source"] = apply_url_source or "api"


def _normalise_remoteok(raw: dict) -> dict | None:
    title = raw.get("position")
    company = raw.get("company")
    url = normalize_url(raw.get("url") or "")
    desc = raw.get("description") or ""

    if not all([title, company, url]):
        return None

    desc = ROJOX_PATTERN.sub("", desc).strip()

    date_posted = None
    epoch = raw.get("epoch") or raw.get("date")
    if epoch:
        try:
            date_posted = datetime.utcfromtimestamp(int(epoch)).date().isoformat()
        except (ValueError, TypeError):
            pass

    salary = None
    s_min = raw.get("salary_min")
    s_max = raw.get("salary_max")
    if s_min and s_max:
        salary = f"${s_min:,} - ${s_max:,}"
    elif s_min:
        salary = f"${s_min:,}+"

    apply_url = parse_api_apply_url(raw, normalize=normalize_url)

    job = _empty_job()
    job["title"] = title
    job["company"] = company
    job["description"] = desc
    _set_posting_urls(
        job,
        posting_url=url,
        apply_url=apply_url,
        apply_url_source="api" if apply_url else None,
    )
    job["location"] = raw.get("location")
    job["tags"] = list(raw.get("tags") or [])
    job["salary"] = salary
    job["date_posted"] = date_posted
    job["source"] = "remoteok"
    return job


def _normalise_remotive(raw: dict) -> dict | None:
    title = raw.get("title")
    company = raw.get("company_name")
    url = normalize_url(raw.get("url") or "")
    desc = raw.get("description") or ""

    if not all([title, company, url]):
        return None

    date_posted = None
    pub = raw.get("publication_date")
    if pub:
        try:
            date_posted = datetime.fromisoformat(pub[:10]).date().isoformat()
        except (ValueError, TypeError):
            pass

    job = _empty_job()
    job["title"] = title
    job["company"] = company
    job["description"] = desc
    _set_posting_urls(job, posting_url=url)
    job["location"] = raw.get("candidate_required_location")
    job["tags"] = list(raw.get("tags") or [])
    job["salary"] = raw.get("salary") or None
    job["date_posted"] = date_posted
    job["source"] = "remotive"
    return job


# WeWorkRemotely titles are "Company Name: Job Title" or "Company at Job Title".
_WWR_TITLE_SPLIT = re.compile(r"\s+(?::|at)\s+", re.IGNORECASE)


def _normalise_wwr(item: ET.Element) -> dict | None:
    title_el = item.find("title")
    link_el = item.find("link")
    desc_el = item.find("description")
    pub_el = item.find("pubDate")
    region_el = item.find("region")

    title_raw = (title_el.text if title_el is not None else "") or ""
    url = normalize_url((link_el.text if link_el is not None else "") or "")
    desc = (desc_el.text if desc_el is not None else "") or ""

    if not title_raw or not url:
        return None

    parts = _WWR_TITLE_SPLIT.split(title_raw, maxsplit=1)
    if len(parts) == 2:
        company, title = parts[0].strip(), parts[1].strip()
    else:
        company, title = None, title_raw.strip()

    if not title or not company:
        return None

    date_posted = None
    pub = (pub_el.text if pub_el is not None else "") or ""
    if pub:
        for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z"):
            try:
                date_posted = datetime.strptime(pub, fmt).date().isoformat()
                break
            except (ValueError, TypeError):
                continue

    job = _empty_job()
    job["title"] = title
    job["company"] = company
    job["description"] = desc
    job["url"] = url
    job["location"] = (region_el.text if region_el is not None else None)
    job["tags"] = []
    job["salary"] = None
    job["date_posted"] = date_posted
    job["source"] = "weworkremotely"
    return job


def _himalayas_posting_url(raw: dict) -> str:
    listing = normalize_url(raw.get("url") or raw.get("link") or "")
    if listing:
        return listing
    guid = raw.get("guid")
    if guid:
        return normalize_url(f"https://himalayas.app/jobs/{guid}")
    return ""


def _normalise_himalayas(raw: dict) -> dict | None:
    title = raw.get("title")
    company = raw.get("companyName") or raw.get("company_name")
    posting_url = _himalayas_posting_url(raw)
    apply_url = normalize_url(raw.get("applicationLink") or "") or None
    desc = (
        raw.get("fullText")
        or raw.get("description")
        or raw.get("excerpt")
        or ""
    )

    if not all([title, company, posting_url]):
        return None

    date_posted = None
    pub = raw.get("pubDate") or raw.get("publishedAt") or raw.get("publication_date")
    if pub:
        try:
            date_posted = datetime.fromisoformat(str(pub)[:10]).date().isoformat()
        except (ValueError, TypeError):
            pass

    location = None
    if isinstance(raw.get("locationRestrictions"), list) and raw["locationRestrictions"]:
        location = ", ".join(str(x) for x in raw["locationRestrictions"])

    job = _empty_job()
    job["title"] = title
    job["company"] = company
    job["description"] = desc
    _set_posting_urls(
        job,
        posting_url=posting_url,
        apply_url=apply_url,
        apply_url_source="api" if apply_url else None,
    )
    job["location"] = location
    job["tags"] = list(raw.get("categories") or [])
    job["salary"] = raw.get("salary") or None
    job["date_posted"] = date_posted
    job["source"] = "himalayas"
    return job


def _normalise_arbeitnow(raw: dict) -> dict | None:
    title = raw.get("title")
    company = raw.get("company_name")
    url = normalize_url(raw.get("url") or "")
    desc = raw.get("description") or ""

    if not all([title, company, url]):
        return None

    date_posted = None
    created = raw.get("created_at")
    if created:
        try:
            date_posted = datetime.utcfromtimestamp(int(created)).date().isoformat()
        except (ValueError, TypeError):
            try:
                date_posted = datetime.fromisoformat(str(created)[:10]).date().isoformat()
            except (ValueError, TypeError):
                pass

    job = _empty_job()
    job["title"] = title
    job["company"] = company
    job["description"] = desc
    job["url"] = url
    job["location"] = raw.get("location")
    job["tags"] = list(raw.get("tags") or [])
    job["salary"] = None
    job["date_posted"] = date_posted
    job["source"] = "arbeitnow"
    return job


# ---------------------------------------------------------------------------
# Remotive apply URL enrichment (detail page HTML)
# ---------------------------------------------------------------------------

async def _fetch_remotive_apply_url(
    client: httpx.AsyncClient,
    posting_url: str,
) -> str | None:
    """Best-effort parse of Remotive listing page for external apply link."""
    try:
        response = await client.get(
            posting_url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
            follow_redirects=True,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for anchor in soup.find_all("a", href=True):
            classes = " ".join(anchor.get("class") or []).lower()
            text = (anchor.get_text() or "").strip().lower()
            if "apply" in classes or text in ("apply", "apply now", "apply for this job"):
                href = anchor["href"].strip()
                if href and not href.startswith("#"):
                    return normalize_url(urljoin(posting_url, href))
    except Exception as exc:
        logger.debug("Remotive apply URL fetch failed for %s: %s", posting_url, exc)
    return None


async def _enrich_remotive_apply_urls(
    client: httpx.AsyncClient,
    jobs: list[dict],
) -> None:
    """Fetch Remotive detail pages for jobs missing apply_url (rate-limited)."""
    targets = [j for j in jobs if j.get("source") == "remotive" and not j.get("apply_url")]
    if not targets:
        return

    sem = asyncio.Semaphore(REMOTIVE_ENRICH_CONCURRENCY)

    async def _one(job: dict) -> None:
        async with sem:
            apply = await _fetch_remotive_apply_url(client, job["url"])
            if apply:
                job["apply_url"] = apply
                job["apply_url_source"] = "detail_fetch"
            await asyncio.sleep(REMOTIVE_ENRICH_DELAY_SEC)

    print(
        f"[Scraper] Enriching {len(targets)} Remotive jobs for apply_url "
        f"(concurrency={REMOTIVE_ENRICH_CONCURRENCY})...",
        flush=True,
    )
    await asyncio.gather(*[_one(j) for j in targets])


# ---------------------------------------------------------------------------
# Keyword post-filter (used by sources that don't support server-side search)
# ---------------------------------------------------------------------------

def _job_haystack(job: dict) -> str:
    """Lower-cased title + description + tags blob for keyword matching."""
    title = job.get("title") or ""
    desc  = job.get("description") or ""
    tags  = " ".join(job.get("tags") or [])
    return f"{title}\n{desc}\n{tags}".lower()


def _word_pattern(token: str) -> re.Pattern[str]:
    pat = _WORD_BOUNDARY_CACHE.get(token)
    if pat is None:
        pat = re.compile(rf"\b{re.escape(token)}\b")
        _WORD_BOUNDARY_CACHE[token] = pat
    return pat


def _keyword_present(haystack: str, keyword: str) -> bool:
    """
    True when `keyword` is present in `haystack`.

    Tries cheapest first: exact lowercase substring (catches "machine
    learning engineer" → "...for a Machine Learning Engineer..."). For
    multi-word keywords that don't appear as a phrase, falls back to
    requiring every word-token to appear with word boundaries. Single-word
    keywords don't get the fallback (substring is enough and word
    boundaries make "ml" miss "ml-ops"-style hits).
    """
    kw = keyword.lower().strip()
    if not kw:
        return False
    if kw in haystack:
        return True
    tokens = [t for t in re.split(r"[\s/+,&\-]+", kw) if t]
    if len(tokens) <= 1:
        return False
    return all(_word_pattern(t).search(haystack) for t in tokens)


def _keyword_match(job: dict, keywords: list[str]) -> bool:
    """OR semantics: a job passes if any user keyword is present."""
    if not keywords:
        return True
    haystack = _job_haystack(job)
    return any(_keyword_present(haystack, kw) for kw in keywords)


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

async def _fetch_remoteok(
    client: httpx.AsyncClient,
    keywords: list[str] | None = None,
) -> list[dict]:
    try:
        response = await client.get(
            REMOTEOK_URL,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        logger.warning("RemoteOK fetch failed: %s", exc)
        return []

    global _REMOTEOK_SAMPLE_LOGGED
    if not _REMOTEOK_SAMPLE_LOGGED:
        sample = next(
            (item for item in data if isinstance(item, dict) and item.get("position")),
            None,
        )
        if sample:
            logger.info(
                "RemoteOK sample job keys: %s",
                sorted(sample.keys()),
            )
            print(
                "[Scraper] RemoteOK sample job (field discovery):\n"
                f"{json.dumps(sample, indent=2, default=str)[:4000]}",
                flush=True,
            )
        _REMOTEOK_SAMPLE_LOGGED = True

    raw_jobs = [
        item for item in data if isinstance(item, dict) and item.get("position")
    ]
    results = []
    for raw in raw_jobs:
        job = _normalise_remoteok(raw)
        if job:
            results.append(job)
    if keywords:
        before = len(results)
        results = [j for j in results if _keyword_match(j, keywords)]
        logger.info(
            "RemoteOK keyword post-filter: %d -> %d", before, len(results)
        )
    results.sort(key=lambda j: j.get("date_posted") or "", reverse=True)
    return dedupe_jobs_merge_apply(results, normalize=normalize_url)


async def _fetch_remotive(
    client: httpx.AsyncClient,
    keywords: list[str] | None = None,
) -> list[dict]:
    """
    Remotive supports server-side search via `?search=`. When keywords are
    provided we fan out one query per keyword in parallel and union the
    results; without keywords we fall back to a single broad fetch.
    """
    if keywords:
        params_list = [{"search": kw, "limit": 50} for kw in keywords]
    else:
        params_list = [{"limit": 100}]

    async def _one(params: dict) -> list[dict]:
        try:
            response = await client.get(REMOTIVE_URL, params=params, timeout=15)
            response.raise_for_status()
            return response.json().get("jobs", [])
        except Exception as exc:
            logger.warning("Remotive fetch failed for params=%s: %s", params, exc)
            return []

    raw_lists = await asyncio.gather(*[_one(p) for p in params_list])

    seen: set[str] = set()
    results = []
    for raw_jobs in raw_lists:
        for raw in raw_jobs:
            job = _normalise_remotive(raw)
            if not job:
                continue
            if job["url"] in seen:
                continue
            seen.add(job["url"])
            results.append(job)
    results.sort(key=lambda j: j.get("date_posted") or "", reverse=True)
    return dedupe_jobs_merge_apply(results, normalize=normalize_url)


async def _fetch_weworkremotely(
    client: httpx.AsyncClient,
    keywords: list[str] | None = None,
) -> list[dict]:
    try:
        response = await client.get(
            WEWORKREMOTELY_RSS,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
    except Exception as exc:
        logger.warning("WeWorkRemotely RSS fetch failed: %s", exc)
        return []

    items = root.findall(".//item")
    results = []
    seen: set[str] = set()
    for item in items:
        try:
            job = _normalise_wwr(item)
        except Exception as exc:
            logger.debug("WWR item parse failed: %s", exc)
            continue
        if not job:
            continue
        if job["url"] in seen:
            continue
        seen.add(job["url"])
        results.append(job)
    if keywords:
        before = len(results)
        results = [j for j in results if _keyword_match(j, keywords)]
        logger.info(
            "WeWorkRemotely keyword post-filter: %d -> %d", before, len(results)
        )
    results.sort(key=lambda j: j.get("date_posted") or "", reverse=True)
    return dedupe_jobs_merge_apply(results, normalize=normalize_url)


async def _fetch_himalayas(
    client: httpx.AsyncClient,
    keywords: list[str] | None = None,
) -> list[dict]:
    try:
        response = await client.get(
            HIMALAYAS_URL,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.warning("Himalayas fetch failed: %s", exc)
        return []

    raw_jobs = payload.get("jobs") if isinstance(payload, dict) else payload
    if not isinstance(raw_jobs, list):
        return []

    results = []
    seen: set[str] = set()
    for raw in raw_jobs:
        if not isinstance(raw, dict):
            continue
        job = _normalise_himalayas(raw)
        if not job:
            continue
        if job["url"] in seen:
            continue
        seen.add(job["url"])
        results.append(job)
    if keywords:
        before = len(results)
        results = [j for j in results if _keyword_match(j, keywords)]
        logger.info(
            "Himalayas keyword post-filter: %d -> %d", before, len(results)
        )
    results.sort(key=lambda j: j.get("date_posted") or "", reverse=True)
    return dedupe_jobs_merge_apply(results, normalize=normalize_url)


async def _fetch_arbeitnow(client: httpx.AsyncClient, terms: list[str]) -> list[dict]:
    """
    Arbeitnow's API requires a search term (empty list returns nothing). The
    caller chooses what to pass: user signal keywords for keyword-driven mode,
    or DEFAULT_SECONDARY_TERMS for the broad-firehose pass.

    We fan out per term in parallel so 6 keywords don't add 6 sequential
    HTTP round-trips.
    """
    if not terms:
        return []

    async def _one(term: str) -> list[dict]:
        try:
            response = await client.get(
                ARBEITNOW_URL,
                params={"search": term},
                timeout=15,
            )
            response.raise_for_status()
            return response.json().get("data", [])
        except Exception as exc:
            logger.warning("Arbeitnow fetch failed for term '%s': %s", term, exc)
            return []

    raw_lists = await asyncio.gather(*[_one(t) for t in terms])

    results = []
    for raw_jobs in raw_lists:
        for raw in raw_jobs:
            job = _normalise_arbeitnow(raw)
            if job:
                results.append(job)
    return dedupe_jobs_merge_apply(results, normalize=normalize_url)


def _round_robin(*sources: list[dict]):
    """Yield items from each source in turn so one source can't dominate."""
    for tup in zip_longest(*sources, fillvalue=None):
        for item in tup:
            if item is not None:
                yield item


# ---------------------------------------------------------------------------
# Candidate aggregation (used by JobScraper.scrape_into_queue)
# ---------------------------------------------------------------------------

async def _collect_candidates(
    client: httpx.AsyncClient,
    keywords: list[str],
) -> list[dict]:
    """
    Hit all five sources in parallel for a given keyword set and return a
    deduped, source-interleaved union.

    Empty keywords -> broad-firehose mode. Arbeitnow still needs *some*
    search term to return data, so we hand it DEFAULT_SECONDARY_TERMS in
    that case.
    """
    arbeitnow_terms = list(keywords) if keywords else list(DEFAULT_SECONDARY_TERMS)

    (
        remoteok_jobs,
        remotive_jobs,
        wwr_jobs,
        himalayas_jobs,
        arbeitnow_jobs,
    ) = await asyncio.gather(
        _fetch_remoteok(client, keywords),
        _fetch_remotive(client, keywords),
        _fetch_weworkremotely(client, keywords),
        _fetch_himalayas(client, keywords),
        _fetch_arbeitnow(client, arbeitnow_terms),
    )

    logger.info(
        "Candidate counts (keywords=%s) — RemoteOK=%d Remotive=%d WWR=%d "
        "Himalayas=%d Arbeitnow=%d",
        keywords or "<broad>",
        len(remoteok_jobs),
        len(remotive_jobs),
        len(wwr_jobs),
        len(himalayas_jobs),
        len(arbeitnow_jobs),
    )

    seen: set[str] = set()
    union: list[dict] = []
    for job in _round_robin(
        remoteok_jobs,
        remotive_jobs,
        wwr_jobs,
        himalayas_jobs,
        arbeitnow_jobs,
    ):
        url = normalize_url(job.get("url") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        union.append(job)
    return union


def _merge_dedupe(existing: list[dict], extra: list[dict]) -> list[dict]:
    """Append `extra` onto `existing`, skipping duplicates by URL."""
    seen: set[str] = {
        normalize_url(j.get("url") or "") for j in existing if j.get("url")
    }
    seen.discard("")
    out = list(existing)
    for job in extra:
        url = normalize_url(job.get("url") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(job)
    return out


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class JobScraper:
    """
    Producer: scrapes remote job boards and pushes normalised jobs onto a queue.

    Keyword-driven scraping with pre-pipeline relaxation:
      - Pass 1: fetch every source with the user's full search_keywords
                (Remotive + Arbeitnow use server-side search; RemoteOK,
                 WWR, and Himalayas are post-filtered by keyword overlap).
      - Pass 2: if the deduped candidate pool is < MIN_CANDIDATES and the
                user supplied more than TOP_K_KEYWORDS keywords, retry
                with the top-K most-general keywords and merge.
      - Pass 3: if still < MIN_CANDIDATES, do an unfiltered firehose pass
                and merge.

    All passes happen before any prefilter / extraction / LLM cost is
    incurred — so relaxation is cheap. Pass stop_event to end emission
    early when downstream has accepted enough jobs.
    """

    async def scrape_into_queue(
        self,
        profile: dict,
        queue: asyncio.Queue,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        # Read keywords off the profile (signal_profile.search_keywords).
        # Defensively coerce to a clean list of stripped strings.
        raw_keywords = []
        if isinstance(profile, dict):
            raw_keywords = profile.get("search_keywords") or []
        keywords: list[str] = [
            k.strip() for k in raw_keywords
            if isinstance(k, str) and k.strip()
        ]

        print(
            f"[Scraper] Starting scrape — keywords={keywords or '<broad firehose>'}",
            flush=True,
        )
        seen_by_url: dict[str, dict] = {}
        stopped_early = False

        def _should_stop() -> bool:
            return stop_event is not None and stop_event.is_set()

        async def emit(job: dict | None) -> bool:
            nonlocal stopped_early
            if _should_stop():
                stopped_early = True
                return True
            if not job:
                return False
            url = normalize_url(job.get("url") or "")
            if not url:
                return False
            if url in seen_by_url:
                merge_apply_url(seen_by_url[url], job)
                return False
            seen_by_url[url] = job
            job["url"] = url
            apply_hint = (
                f" apply={job.get('apply_url')!r}"
                if job.get("apply_url")
                else ""
            )
            await queue.put(job)
            print(
                f"[Scraper] Enqueued ({job.get('source')}): {job.get('title')} @ {url}"
                f"{apply_hint}",
                flush=True,
            )
            return _should_stop()

        try:
            async with httpx.AsyncClient() as client:
                # Pass 1 — strict keyword-driven fetch.
                print(
                    f"[Scraper] Pass 1 — fetching with {len(keywords)} keyword(s)",
                    flush=True,
                )
                candidates = await _collect_candidates(client, keywords)
                print(
                    f"[Scraper] Pass 1 produced {len(candidates)} unique candidates",
                    flush=True,
                )

                # Pass 2 — top-K keywords (only when keywords were provided
                # AND we have more than K of them to actually trim).
                if (
                    len(candidates) < MIN_CANDIDATES
                    and len(keywords) > TOP_K_KEYWORDS
                ):
                    top = keywords[:TOP_K_KEYWORDS]
                    print(
                        f"[Scraper] Pass 2 — pool < {MIN_CANDIDATES}; "
                        f"relaxing to top {TOP_K_KEYWORDS} keywords: {top}",
                        flush=True,
                    )
                    extra = await _collect_candidates(client, top)
                    candidates = _merge_dedupe(candidates, extra)
                    print(
                        f"[Scraper] After Pass 2: {len(candidates)} candidates",
                        flush=True,
                    )

                # Pass 3 — broad firehose. Only when keywords were used
                # in earlier passes; otherwise Pass 1 was already broad
                # and another unfiltered call would just be a duplicate.
                if len(candidates) < MIN_CANDIDATES and keywords:
                    print(
                        f"[Scraper] Pass 3 — pool still < {MIN_CANDIDATES}; "
                        "relaxing to unfiltered firehose",
                        flush=True,
                    )
                    extra = await _collect_candidates(client, [])
                    candidates = _merge_dedupe(candidates, extra)
                    print(
                        f"[Scraper] After Pass 3: {len(candidates)} candidates",
                        flush=True,
                    )

                # Best-effort apply_url enrichment for Remotive (HTTP only;
                # cheap compared to the LLM stages downstream).
                await _enrich_remotive_apply_urls(client, candidates)

                # Emit candidates one at a time. emit() short-circuits if
                # stop_event was set by the streaming consumer.
                for job in candidates:
                    if await emit(job):
                        break

            logger.info(
                "Scraper enqueued %d unique jobs%s.",
                len(seen_by_url),
                " (stopped early)" if stopped_early or _should_stop() else "",
            )
            print(
                f"[Scraper] Finished — {len(seen_by_url)} unique jobs enqueued"
                f"{' (stopped early)' if stopped_early or _should_stop() else ''}",
                flush=True,
            )
        finally:
            print("[Scraper] Sending QUEUE_END to work queue", flush=True)
            await queue.put(QUEUE_END)
