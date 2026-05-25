"""
Rule-based scam detection. Replaces the BERT scorer.

Each rule returns a flag of the form:
    {"level": "auto_fail" | "warning", "message": "..."}

A single auto_fail flag is sufficient to set the legitimacy status to fraud.
Warning flags merely deduct points from the composite score.
"""

import re

from parsing.apply_urls import FREE_APPLY_HOSTS, host_from_url

# --- Patterns -------------------------------------------------------------

PAYMENT_PATTERNS = [
    r"\bregistration fee\b",
    r"\bprocessing fee\b",
    r"\bapplication fee\b",
    r"\bpay (?:a |the )?(?:fee|deposit)\b",
    r"\bsend\s+\$?\d",
    r"\bwestern union\b",
    r"\bmoneygram\b",
    r"\bgift card",
    r"\bbitcoin\b.*\b(?:to apply|deposit|fee)\b",
]

CONTACT_OFFSITE_PATTERNS = [
    r"\b(?:contact|message|text|whatsapp|telegram|signal)\b.*\b(?:\+?\d[\d\s\-()]{7,})",
    r"\bwhatsapp\b",
    r"\btelegram\b",
    r"\bsignal\s+(?:me|us|app)\b",
    r"\bemail\s+me\s+at\b.*@(?:gmail|yahoo|outlook|hotmail|protonmail|aol)\.",
]

IDENTITY_THEFT_PATTERNS = [
    r"\bsocial security number\b",
    r"\bssn\b(?!\s*(?:not|won['’]t))",
    r"\bbank account (?:number|details|info)\b",
    r"\brouting number\b",
    r"\bcredit card (?:number|details)\b",
    r"\bcopy of (?:your )?(?:passport|driver['’]s? license|id)\b",
    r"\bid (?:photo|picture|scan)\b",
]

SUSPICIOUS_TITLE_PATTERNS = [
    r"\bdata entry\b.*\$\s*\d{2,3}\s*/\s*(?:hr|hour)",
    r"\bmystery shopper\b",
    r"\bsecret shopper\b",
    r"\bpackage\s+forwarder\b",
    r"\benvelope\s+stuffer\b",
    r"\bmoney mule\b",
    r"\bpersonal assistant\b.*\$\s*\d{2,3}\s*/\s*(?:hr|hour)",
    r"\bearn\s+\$\d{2,3}\s*/\s*(?:hr|hour)\s+from\s+home\b",
]

# Free-email-as-employer-contact when the company *should* have a domain.
FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
    "protonmail.com", "aol.com", "icloud.com", "live.com",
    "yandex.com", "mail.com",
}

# Heuristic: roles whose advertised pay implies experience requirements.
SENIOR_ROLE_RX = re.compile(
    r"\b(senior|staff|principal|lead|director|vp|head of)\b", re.IGNORECASE
)

KNOWN_ATS_HOST_SUFFIXES = (
    "greenhouse.io",
    "lever.co",
    "ashbyhq.com",
    "workable.com",
    "breezy.hr",
    "recruitee.com",
    "smartrecruiters.com",
    "job-boards.greenhouse.io",
    "jobs.lever.co",
    "apply.workable.com",
)


# --- Helpers --------------------------------------------------------------

def _hits(patterns: list[str], text: str) -> list[str]:
    return [p for p in patterns if re.search(p, text, flags=re.IGNORECASE)]


def _flag(level: str, message: str) -> dict:
    return {"level": level, "message": message}


def _is_ats_host(host: str) -> bool:
    return any(
        host == suffix or host.endswith(f".{suffix}")
        for suffix in KNOWN_ATS_HOST_SUFFIXES
    )


def _domain_related(apply_host: str, company_domain: str) -> bool:
    if apply_host == company_domain or apply_host.endswith(f".{company_domain}"):
        return True
    return _is_ats_host(apply_host)


def _apply_url_flags(extracted: dict) -> list[dict]:
    """Extra scrutiny when apply URL came from LLM extraction, not API/HTML."""
    apply_url = extracted.get("apply_url")
    source = extracted.get("apply_url_source")
    if not apply_url or source != "extractor":
        return []

    flags: list[dict] = []
    host = host_from_url(apply_url) or ""
    company_domain = (extracted.get("company_domain") or "").lower().strip()

    if host in FREE_APPLY_HOSTS:
        flags.append(_flag(
            "warning",
            f"Extractor-sourced apply URL uses personal/forms host ({host}).",
        ))
    elif company_domain and host and not _domain_related(host, company_domain):
        flags.append(_flag(
            "warning",
            f"Extractor-sourced apply URL host ({host}) does not match "
            f"company domain ({company_domain}).",
        ))
    return flags


def _try_int(v) -> int | None:
    if v is None:
        return None
    try:
        return int(str(v).replace(",", "").strip() or "0")
    except (ValueError, TypeError):
        return None


# --- Main entry -----------------------------------------------------------

def rule_based_scam_check(extracted: dict, raw_text: str) -> list[dict]:
    """
    Inspect the extractor output + raw posting text and return a list of
    flags. Caller (legitimacy_check) decides status from these flags plus
    company verification.
    """
    flags: list[dict] = []
    text_blob = " ".join([
        raw_text or "",
        extracted.get("job_description") or "",
        " ".join(extracted.get("responsibilities") or []),
        " ".join(extracted.get("required_qualifications") or []),
    ])

    if _hits(PAYMENT_PATTERNS, text_blob):
        flags.append(_flag(
            "auto_fail",
            "Posting requests payment, fees, or deposits from applicant.",
        ))

    if _hits(IDENTITY_THEFT_PATTERNS, text_blob):
        flags.append(_flag(
            "auto_fail",
            "Posting demands SSN, bank, or government-ID details up front.",
        ))

    if _hits(SUSPICIOUS_TITLE_PATTERNS, text_blob):
        flags.append(_flag(
            "auto_fail",
            "Title/body match a known scam template (mule, mystery shopper, etc.).",
        ))

    if _hits(CONTACT_OFFSITE_PATTERNS, text_blob):
        flags.append(_flag(
            "warning",
            "Posting routes communication off-platform (WhatsApp/Telegram/personal email).",
        ))

    contact_email = (extracted.get("contact_email") or "").lower().strip()
    company_domain = (extracted.get("company_domain") or "").lower().strip()
    if contact_email and "@" in contact_email:
        local, _, domain = contact_email.partition("@")
        if domain in FREE_EMAIL_DOMAINS and company_domain:
            flags.append(_flag(
                "warning",
                f"Employer contact uses free-email ({domain}) but a company "
                f"domain ({company_domain}) exists.",
            ))

    salary_min = _try_int(extracted.get("salary_min"))
    period = (extracted.get("salary_period") or "").lower()
    title = extracted.get("job_title") or ""
    if salary_min and period in ("hourly", "weekly") and salary_min >= 80 and not SENIOR_ROLE_RX.search(title):
        # Example: "$80/hr data entry, no experience required" with non-senior title.
        if "no experience" in text_blob.lower() or "entry-level" in text_blob.lower():
            flags.append(_flag(
                "warning",
                "Implausibly high pay for stated experience level.",
            ))

    flags.extend(_apply_url_flags(extracted))
    return flags
