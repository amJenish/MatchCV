"""Shared job text rules used by prefilter (and optionally other stages)."""

import re

INPERSON_PHRASES = [
    "must be on-site",
    "in-office required",
    "no remote",
    "onsite only",
    "on-site only",
    "in person only",
    "in-person only",
    "relocation required",
    "must relocate",
    "office based",
    "office-based",
    "not a remote position",
    "this is not a remote",
]

# Substrings that mark a posting as a stub / aggregator preview
# rather than a substantive listing. Hit on any of these and we drop
# before reaching the LLM stages.
STUB_MARKERS = [
    "see this and similar jobs on linkedin",
    "veja esta vaga e outras semelhantes",
    "anuncio publicado",
    "anunciada ",
]

# Lead-in patterns that appear at the start of LinkedIn-reposted stubs.
STUB_PREFIX_PATTERNS = [
    re.compile(r"^posted\s+\d{1,2}:\d{2}", re.IGNORECASE),
    re.compile(r"^anunciada\s+\d{1,2}:\d{2}", re.IGNORECASE),
]

MAX_POSTING_AGE_DAYS = 40
MIN_DESC_LENGTH = 400


def is_explicitly_inperson(title: str, description: str, location: str = "") -> bool:
    """
    True only when the posting explicitly requires in-person work.
    Ambiguous or unstated remote status -> False (treat as remote-eligible).
    """
    haystack = f"{title} {description} {location}".lower()
    return any(phrase in haystack for phrase in INPERSON_PHRASES)


def is_stub(title: str, description: str) -> bool:
    """
    True when the posting body looks like an aggregator stub or a
    LinkedIn redirect preview rather than the real listing.
    """
    body = (description or "").lower()
    if any(marker in body for marker in STUB_MARKERS):
        return True
    body_head = body.lstrip()[:80]
    if any(p.match(body_head) for p in STUB_PREFIX_PATTERNS):
        # Stubs are short. A real posting that happens to start with
        # "Posted 12:34" wouldn't be under MIN_DESC_LENGTH.
        if len(body) < MIN_DESC_LENGTH:
            return True
    return False
