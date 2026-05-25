import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


QUALITY_RUBRIC = """
You score a remote job posting on 6 dimensions, 0-10 each, then flag penalty caps.
Output strictly via the save_quality tool. Never write prose.

DIMENSIONS (0-10 each, anchors at 0/4/7/10)

1. compensation
   10  Numeric range stated with currency and period (e.g. "$80k-$110k/year").
    7  Single number stated with period and currency.
    4  Range stated with no currency, OR "DOE" with a target band.
    0  No compensation, "Competitive", "TBD", or empty.

2. role_definition
   10  4+ specific responsibilities listed; reader can describe day-to-day.
    7  3 specific responsibilities listed.
    4  Generic responsibilities ("collaborate with team") only.
    0  No responsibilities listed, or "wear many hats" with nothing concrete.

3. requirement_proportionality
   10  Required qualifications match seniority claimed by title and pay band.
    7  Slight overreach (one senior skill listed for a mid role) but reasonable.
    4  Material overreach (senior experience demanded for junior pay).
    0  Demands FAANG/PhD experience for entry pay; or no requirements stated.

4. company_identity
   10  Company named, has a verifiable web presence in the posting (URL/about-us/domain).
    7  Company named with no link, but unambiguous.
    4  Company named only by slug or domain; no explicit name in body.
    0  "Confidential", recruiter-fronted with hidden client, or no company at all.

5. description_substance
   10  Multi-paragraph description of role, team, company context.
    7  Solid paragraph with role + team or role + product context.
    4  Single short paragraph; mostly marketing.
    0  Less than 200 chars of substantive text; or stub/redirect.

6. structural_completeness
   10  Has all of: title, company, responsibilities, requirements, location/remote info, comp.
    7  Missing one of those.
    4  Missing two of those.
    0  Missing three or more.

PENALTY CAPS (set true to cap the resulting total score)

- truncation_or_stub: total capped at 15 if the posting is a redirect/aggregator preview.
- vague_only:         total capped at 25 if every responsibility is generic marketing language.
- title_body_mismatch: total capped at 30 if title and body describe different roles.

Quality is about substance and clarity. It is NOT scam detection.
Do not penalize a posting here for the application channel or "deceptive" language;
those are evaluated by a separate legitimacy stage.

Always include up to 5 short failures and up to 5 short highlights, plus a one-sentence summary.
"""


SAVE_QUALITY_TOOL = {
    "name": "save_quality",
    "description": "Save the structured quality assessment for a job posting.",
    "input_schema": {
        "type": "object",
        "required": ["scores", "penalties", "failures", "highlights", "summary"],
        "properties": {
            "scores": {
                "type": "object",
                "required": [
                    "compensation",
                    "role_definition",
                    "requirement_proportionality",
                    "company_identity",
                    "description_substance",
                    "structural_completeness",
                ],
                "properties": {
                    "compensation":                {"type": "integer", "minimum": 0, "maximum": 10},
                    "role_definition":             {"type": "integer", "minimum": 0, "maximum": 10},
                    "requirement_proportionality": {"type": "integer", "minimum": 0, "maximum": 10},
                    "company_identity":            {"type": "integer", "minimum": 0, "maximum": 10},
                    "description_substance":       {"type": "integer", "minimum": 0, "maximum": 10},
                    "structural_completeness":     {"type": "integer", "minimum": 0, "maximum": 10},
                },
            },
            "penalties": {
                "type": "object",
                "required": ["truncation_or_stub", "vague_only", "title_body_mismatch"],
                "properties": {
                    "truncation_or_stub":  {"type": "boolean"},
                    "vague_only":          {"type": "boolean"},
                    "title_body_mismatch": {"type": "boolean"},
                },
            },
            "failures":   {"type": "array", "items": {"type": "string"}},
            "highlights": {"type": "array", "items": {"type": "string"}},
            "summary":    {"type": "string"},
        },
    },
}


def compute_quality_score(result: dict) -> int:
    """
    Deterministic 0-100 score from rubric components and penalty caps.

    Sum the six 0-10 dimensions, rescale 0-60 -> 0-100, then apply caps.
    """
    s = result.get("scores") or {}
    keys = (
        "compensation",
        "role_definition",
        "requirement_proportionality",
        "company_identity",
        "description_substance",
        "structural_completeness",
    )
    base = sum(int(s.get(k, 0)) for k in keys) * 100 // 60
    base = max(0, min(100, base))

    p = result.get("penalties") or {}
    if p.get("truncation_or_stub"):
        base = min(base, 15)
    if p.get("vague_only"):
        base = min(base, 25)
    if p.get("title_body_mismatch"):
        base = min(base, 30)
    return base


def meets_standard(result: dict, score: int) -> bool:
    """
    True iff the posting passes overall AND the dimensions an applicant
    most needs (compensation visibility, identifiable company, defined role)
    are above their minimums.
    """
    s = result.get("scores") or {}
    p = result.get("penalties") or {}
    if any(p.values()):
        return False
    if score < 70:
        return False
    return (
        int(s.get("compensation", 0))     >= 4
        and int(s.get("company_identity", 0)) >= 7
        and int(s.get("role_definition", 0))  >= 7
    )


async def assess_job_quality(extracted: dict) -> dict:
    """
    Score posting quality using the rubric. extracted is the JobExtractor
    output; we feed the structured fields plus the raw_text so the model
    can apply the dimensions consistently.
    """
    title = extracted.get("job_title") or "?"
    print(f"[Quality] Calling Claude Haiku — title={title!r}", flush=True)

    user_payload = (
        "Score this posting against the rubric.\n\n"
        f"job_title: {extracted.get('job_title')!r}\n"
        f"company_name: {extracted.get('company_name')!r}\n"
        f"company_domain: {extracted.get('company_domain')!r}\n"
        f"is_remote: {extracted.get('is_remote')!r}\n"
        f"salary_stated: {extracted.get('salary_stated')!r}\n"
        f"salary: {extracted.get('salary_min')!r} - {extracted.get('salary_max')!r} "
        f"{extracted.get('salary_currency')!r} {extracted.get('salary_period')!r}\n"
        f"responsibilities: {extracted.get('responsibilities') or []}\n"
        f"required_qualifications: {extracted.get('required_qualifications') or []}\n"
        f"preferred_qualifications: {extracted.get('preferred_qualifications') or []}\n"
        f"benefits: {extracted.get('benefits') or []}\n"
        f"\nFull posting text:\n{extracted.get('raw_text') or ''}"
    )

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        system=QUALITY_RUBRIC,
        tools=[SAVE_QUALITY_TOOL],
        tool_choice={"type": "tool", "name": "save_quality"},
        messages=[{"role": "user", "content": user_payload}],
    )

    raw = _extract_tool_input(message)
    score = compute_quality_score(raw)
    standard = meets_standard(raw, score)

    print(
        f"[Quality] Done — score={score}, meets_standard={standard}, "
        f"penalties={raw.get('penalties')}",
        flush=True,
    )

    return {
        "quality_score": score,
        "meets_standard": standard,
        "quality_components": raw.get("scores", {}),
        "quality_penalties": raw.get("penalties", {}),
        "quality_failures": raw.get("failures", []),
        "quality_highlights": raw.get("highlights", []),
        "quality_summary": raw.get("summary", ""),
    }


def _extract_tool_input(message) -> dict:
    for block in message.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "save_quality":
            return dict(block.input)
    raise RuntimeError(
        "Quality: model did not return a save_quality tool_use block. "
        f"stop_reason={message.stop_reason!r}"
    )
