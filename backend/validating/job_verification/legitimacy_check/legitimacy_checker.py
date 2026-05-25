from urllib.parse import urlsplit

from validating.job_verification.legitimacy_check.clearbit import (
    clearbit_suggestions,
)
from validating.job_verification.legitimacy_check.rules import (
    rule_based_scam_check,
)


def _strip_domain(value: str | None) -> str | None:
    if not value:
        return None
    s = value.strip().lower()
    if not s:
        return None
    if "://" in s:
        s = urlsplit(s).netloc or s
    s = s.split("/", 1)[0]
    s = s.lstrip("www.")
    return s or None


def _normalize_name(name: str | None) -> str:
    return (name or "").strip().lower()


async def verify_company(
    name: str | None,
    claimed_domain: str | None,
) -> dict:
    """
    Resolve a company name + (optional) claimed domain to a verification
    verdict using the full Clearbit suggestion list.

    Reasons:
      - "name_matches_claimed_domain": claimed domain exists in suggestions
        for the same name -> verified.
      - "unique_match": claimed_domain absent but only one suggestion
        whose name matches -> verified.
      - "name_domain_mismatch": claimed domain not present in suggestion
        list for that name -> not verified (records claimed/found).
      - "ambiguous": multiple suggestions and no claimed domain to pick from -> not verified.
      - "no_match": no suggestions returned -> not verified.
      - "no_name": missing/empty company name -> not verified.
    """
    name_norm = _normalize_name(name)
    claimed = _strip_domain(claimed_domain)

    if not name_norm:
        return {
            "verified": False,
            "reason": "no_name",
            "claimed_domain": claimed,
            "matched_domain": None,
            "matched_logo": None,
            "candidates": [],
        }

    suggestions = await clearbit_suggestions(name_norm)
    candidates = [
        {
            "name": s.get("name"),
            "domain": _strip_domain(s.get("domain")),
            "logo": s.get("logo"),
        }
        for s in suggestions
        if s.get("domain")
    ]

    if not candidates:
        return {
            "verified": False,
            "reason": "no_match",
            "claimed_domain": claimed,
            "matched_domain": None,
            "matched_logo": None,
            "candidates": [],
        }

    if claimed:
        for c in candidates:
            if c["domain"] == claimed and _normalize_name(c["name"]).startswith(name_norm[:8]):
                return {
                    "verified": True,
                    "reason": "name_matches_claimed_domain",
                    "claimed_domain": claimed,
                    "matched_domain": c["domain"],
                    "matched_logo": c["logo"],
                    "candidates": candidates,
                }
        return {
            "verified": False,
            "reason": "name_domain_mismatch",
            "claimed_domain": claimed,
            "matched_domain": candidates[0]["domain"],
            "matched_logo": candidates[0]["logo"],
            "candidates": candidates,
        }

    matching = [
        c for c in candidates
        if _normalize_name(c["name"]).startswith(name_norm[:8])
    ]
    if len(matching) == 1:
        return {
            "verified": True,
            "reason": "unique_match",
            "claimed_domain": None,
            "matched_domain": matching[0]["domain"],
            "matched_logo": matching[0]["logo"],
            "candidates": candidates,
        }
    if len(candidates) == 1:
        return {
            "verified": True,
            "reason": "unique_match",
            "claimed_domain": None,
            "matched_domain": candidates[0]["domain"],
            "matched_logo": candidates[0]["logo"],
            "candidates": candidates,
        }
    return {
        "verified": False,
        "reason": "ambiguous",
        "claimed_domain": None,
        "matched_domain": candidates[0]["domain"] if candidates else None,
        "matched_logo": candidates[0]["logo"] if candidates else None,
        "candidates": candidates,
    }


def _composite_legitimacy_score(verification: dict, flags: list[dict]) -> int:
    """
    Deterministic 0-100 score. Start at 100 and deduct.
    auto_fail flags zero the score; the calling status logic then drops the row.
    """
    if any(f.get("level") == "auto_fail" for f in flags):
        return 0

    score = 100
    reason = verification.get("reason")
    if reason == "name_matches_claimed_domain" or reason == "unique_match":
        deduction = 0
    elif reason == "name_domain_mismatch":
        deduction = 30
    elif reason == "ambiguous":
        deduction = 15
    elif reason == "no_match":
        deduction = 25
    elif reason == "no_name":
        deduction = 35
    else:
        deduction = 20
    score -= deduction

    score -= 10 * sum(1 for f in flags if f.get("level") == "warning")
    return max(0, min(100, score))


async def legitimacy_check(extracted: dict, *, scraper_job: dict | None = None) -> dict:
    """
    Returns:
      {
        "status": "legitimate" | "warning" | "fraud",
        "legitimacy_score": int 0-100,
        "flags": [...],
        "company_verification": {...},
      }
    """
    company_name = extracted.get("company_name") or (scraper_job or {}).get("company")
    claimed_domain = extracted.get("company_domain")
    raw_text = extracted.get("raw_text") or (scraper_job or {}).get("description") or ""

    print(f"[Legitimacy] Verifying company: {company_name!r}", flush=True)
    verification = await verify_company(company_name, claimed_domain)
    print(
        f"[Legitimacy] verified={verification.get('verified')} "
        f"reason={verification.get('reason')} "
        f"matched_domain={verification.get('matched_domain')!r}",
        flush=True,
    )

    apply_url = extracted.get("apply_url") or (scraper_job or {}).get("apply_url")
    apply_source = extracted.get("apply_url_source") or (scraper_job or {}).get(
        "apply_url_source"
    )
    if apply_url:
        extracted = {**extracted, "apply_url": apply_url, "apply_url_source": apply_source}

    print("[Legitimacy] Running rule-based scam check", flush=True)
    flags = rule_based_scam_check(extracted, raw_text)
    if flags:
        print(f"[Legitimacy] Flags: {[f['message'] for f in flags]}", flush=True)
    else:
        print("[Legitimacy] No flags raised.", flush=True)

    score = _composite_legitimacy_score(verification, flags)

    if any(f.get("level") == "auto_fail" for f in flags):
        status = "fraud"
    elif score < 35:
        status = "fraud"
    elif score < 70 or any(f.get("level") == "warning" for f in flags) or not verification.get("verified"):
        status = "warning"
    else:
        status = "legitimate"

    print(f"[Legitimacy] status={status} score={score}", flush=True)

    return {
        "status": status,
        "legitimacy_score": score,
        "flags": flags,
        "company_verification": verification,
    }
