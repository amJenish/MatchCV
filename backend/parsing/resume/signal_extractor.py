"""Claude Haiku tool-use: structured resume sections → signal_profile."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)


SAVE_SIGNAL_TOOL = {
    "name": "save_signal_profile",
    "description": (
        "Distill resume sections into the signal profile that downstream job "
        "discovery uses for keyword matching, embedding similarity, and "
        "seniority filtering."
    ),
    "input_schema": {
        "type": "object",
        "required": [
            "suitable_levels",
            "primary_roles",
            "secondary_roles",
            "core_skills",
            "domain_expertise",
            "years_of_experience",
            "seniority_reasoning",
            "search_keywords",
            "deal_breakers",
        ],
        "properties": {
            "suitable_levels": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["Intern", "Junior", "Mid-level", "Senior"],
                },
                "minItems": 1,
                "description": (
                    "Which seniority bands the candidate should target. "
                    "Usually 1-2 adjacent bands."
                ),
            },
            "primary_roles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Job titles the candidate is best suited for. Max 5.",
            },
            "secondary_roles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Plausible adjacent role titles. Max 5.",
            },
            "core_skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Concrete technical skills the resume demonstrates. Max 10.",
            },
            "domain_expertise": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Industries / problem domains the candidate has worked in. Max 5.",
            },
            "years_of_experience": {
                "type": "integer",
                "minimum": 0,
                "maximum": 60,
            },
            "seniority_reasoning": {
                "type": "string",
                "description": "1-2 sentence justification for suitable_levels.",
            },
            "search_keywords": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 6,
                "description": (
                    "Search terms the scraper will hand to job sources. "
                    "Derived from primary_roles + core_skills + domain_expertise, "
                    "deduplicated, max 6. Each term should be specific enough "
                    "to return relevant remote jobs but not so narrow that "
                    "sources return nothing."
                ),
            },
            "deal_breakers": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Things the candidate should avoid: relocation requirements, "
                    "unpaid trials, etc. Empty array when none are inferable."
                ),
            },
        },
    },
}


SYSTEM_PROMPT = """You convert a candidate's structured resume into a job-search signal profile via the save_signal_profile tool. Output strictly via the tool — never write prose.

SENIORITY DEFINITIONS — use scope, ownership, and complexity (not just years):
- Intern: currently studying or zero professional experience.
- Junior: 0–2 years; contributes under supervision; tasks rather than systems.
- Mid-level: 2–5 years; works independently; owns deliverables end-to-end.
- Senior: 5+ years; leads work, mentors others, owns systems not just tasks.

Years alone are not sufficient. A self-led 2-year founder is mid-level, not junior. A 6-year individual contributor with no scope expansion is mid-level, not senior. Use the work_info bullets to judge ownership and complexity.

OUTPUT GUIDANCE
- suitable_levels: usually 1-2 adjacent bands (e.g. ["Junior","Mid-level"] when transitioning). Never include all four.
- primary_roles vs secondary_roles: primary is the highest match; secondary covers adjacent fits. Use job-board canonical titles ("Software Engineer", "Data Scientist", "ML Engineer"), not company-specific titles.
- core_skills: concrete tools, languages, frameworks. Max 10. No soft skills.
- domain_expertise: industries / problem areas (e.g. "fintech", "healthcare ML", "developer tooling"). Max 5.
- search_keywords: 3–6 terms. Build them from primary_roles + core_skills + domain_expertise. Deduplicate and prefer the candidate's strongest combinations. Each keyword is what a job-board search bar would accept (e.g. "machine learning engineer", "python backend", "computer vision"). Avoid single common words like "engineer" alone.
- deal_breakers: only include explicit signals (e.g. resume says "remote-only"). Empty when nothing inferable.
"""


class SignalExtractor:
    """Wrap Anthropic Haiku tool-use to produce the signal profile."""

    def __init__(self) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is missing from backend/.env")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-haiku-4-5-20251001"

    def extract(self, sections: dict) -> dict:
        """
        sections must be the dict the resume_parser produced (or what the
        resume_sections repo returns), with keys:
            work_experience, projects, education, skills
        """
        if not isinstance(sections, dict):
            raise ValueError("SignalExtractor: sections must be a dict")

        compact = {
            "work_experience": sections.get("work_experience") or [],
            "projects":        sections.get("projects") or [],
            "education":       sections.get("education") or [],
            "skills":          sections.get("skills") or [],
        }

        message = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=[SAVE_SIGNAL_TOOL],
            tool_choice={"type": "tool", "name": "save_signal_profile"},
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Resume sections (JSON). Call save_signal_profile.\n\n"
                        + json.dumps(compact, ensure_ascii=False)
                    ),
                }
            ],
        )

        for block in message.content:
            if (
                getattr(block, "type", None) == "tool_use"
                and block.name == "save_signal_profile"
            ):
                profile = dict(block.input)
                # Cap keyword list defensively in case the model overshoots.
                kws = profile.get("search_keywords") or []
                if isinstance(kws, list) and len(kws) > 6:
                    profile["search_keywords"] = kws[:6]
                return profile

        raise RuntimeError(
            "SignalExtractor: model did not return a save_signal_profile tool_use block. "
            f"stop_reason={message.stop_reason!r}"
        )
