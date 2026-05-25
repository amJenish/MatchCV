"""Apply URL provenance, merge precedence, and host helpers."""

from urllib.parse import urlsplit

from parsing.content_hash import compute_content_hash

# Higher number = prefer when merging duplicate postings.
APPLY_SOURCE_TRUST: dict[str | None, int] = {
    "api": 3,
    "detail_fetch": 2,
    "extractor": 1,
    None: 0,
}

FREE_APPLY_HOSTS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
    "protonmail.com", "aol.com", "icloud.com", "live.com",
    "docs.google.com", "forms.gle", "tally.so", "typeform.com",
}


def apply_source_trust(source: str | None) -> int:
    return APPLY_SOURCE_TRUST.get(source, 0)


def host_from_url(url: str | None) -> str | None:
    if not url:
        return None
    try:
        netloc = urlsplit(url.strip()).netloc.lower()
    except ValueError:
        return None
    return netloc.lstrip("www.") or None


def merge_apply_url(dst: dict, src: dict) -> None:
    """
    Prefer the apply_url with higher-trust apply_url_source.
    Fill apply_url on dst when dst lacks one and src has one.
    """
    dst_url = dst.get("apply_url")
    dst_src = dst.get("apply_url_source")
    src_url = src.get("apply_url")
    src_src = src.get("apply_url_source")

    if not src_url:
        return

    if not dst_url:
        dst["apply_url"] = src_url
        dst["apply_url_source"] = src_src
        return

    if apply_source_trust(src_src) > apply_source_trust(dst_src):
        dst["apply_url"] = src_url
        dst["apply_url_source"] = src_src


def parse_api_apply_url(raw: dict, *, normalize) -> str | None:
    """Defensive RemoteOK / generic API apply field discovery."""
    for key in ("apply_url", "applyUrl", "original"):
        val = raw.get(key)
        if val:
            normalized = normalize(str(val))
            if normalized:
                return normalized
    return None


def dedupe_jobs_merge_apply(jobs: list[dict], *, normalize) -> list[dict]:
    """
    Dedupe a source list by posting URL then content hash, merging apply_url
    from duplicates onto the kept record.
    """
    by_url: dict[str, dict] = {}
    for job in jobs:
        url = normalize(job.get("url") or "")
        if not url:
            continue
        job["url"] = url
        if url in by_url:
            merge_apply_url(by_url[url], job)
        else:
            by_url[url] = job

    by_hash: dict[str, dict] = {}
    for job in by_url.values():
        ch = compute_content_hash(job)
        if ch in by_hash:
            merge_apply_url(by_hash[ch], job)
        else:
            by_hash[ch] = job
    return list(by_hash.values())
