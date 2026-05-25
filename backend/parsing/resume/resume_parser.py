"""Claude Haiku tool-use: resume text → structured sections."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)


# Single tool with the four sections so the model commits to one structured
# output. Fields are nullable to allow honest "missing" answers without
# forcing the model to fabricate.
SAVE_RESUME_TOOL = {
    "name": "save_resume",
    "description": (
        "Save the structured contents of a resume. Each section is a list "
        "of items extracted verbatim from the resume. Empty list when absent."
    ),
    "input_schema": {
        "type": "object",
        "required": ["work_experience", "projects", "education", "skills"],
        "properties": {
            "work_experience": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["work_title", "company", "work_start", "is_current"],
                    "properties": {
                        # work_title and company are NOT NULL in DB.
                        "work_title": {"type": "string"},
                        "company":    {"type": "string"},
                        # ISO date YYYY-MM-DD or null. Postgres column is `date`.
                        "work_start": {"type": ["string", "null"]},
                        "work_end":   {"type": ["string", "null"]},
                        "is_current": {"type": "boolean"},
                        "work_info":  {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
            "projects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["project_title"],
                    "properties": {
                        "project_title": {"type": ["string", "null"]},
                        "project_info": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "tech_stack": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
            "education": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "institution":     {"type": ["string", "null"]},
                        "degree":          {"type": ["string", "null"]},
                        "field_of_study":  {"type": ["string", "null"]},
                        # Postgres column is integer.
                        "graduation_year": {"type": ["integer", "null"]},
                    },
                },
            },
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Flat deduplicated list of named skills/tools.",
            },
        },
    },
}


SYSTEM_PROMPT = """You parse remote-job applicant resumes into structured sections via the save_resume tool. Output strictly via the tool — never write prose.

EXTRACTION RULES
1. Pull values verbatim from the resume. Do not paraphrase, embellish, or invent.
2. work_experience — one entry per role:
   - work_title and company are required strings (never null). Skip the entry entirely if a role has neither.
   - work_start and work_end MUST be ISO dates (YYYY-MM-DD) or null:
     * "Jan 2022" / "January 2022" → "2022-01-01"
     * "2022" alone → "2022-01-01"
     * "2022-03" → "2022-03-01"
     * "Present" / "Current" / "Now" / ongoing role → set work_end to null AND is_current=true
     * If no start date is shown at all → leave work_start null
   - is_current is a boolean. True ONLY when the role is currently held; false otherwise.
   - work_info is the bullet list under that role, each bullet a short string.
3. projects: include both standalone and side projects.
   - project_title is the project's name as written.
   - tech_stack is the technologies explicitly mentioned for that project.
   - project_info is the bullet list for the project.
4. education: include each degree / certification.
   - graduation_year is the literal year as an integer (e.g. 2024). Null if not stated.
5. skills: a single deduplicated flat list. Include named tools, languages, frameworks, libraries, and methodologies. Skip soft skills like "team player" or marketing fluff.
6. Empty section ⇒ empty array, never null at the section level. Per-field nulls are fine when a value is genuinely missing.
7. Ignore contact info, summary blurbs, and personal interests — these are not part of the structured sections.
"""


class ResumeParser:
    """Wrap Anthropic Haiku tool-use to extract resume sections."""

    def __init__(self) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is missing from backend/.env")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-haiku-4-5-20251001"

    def parse(self, resume_text: str) -> dict:
        if not resume_text or not resume_text.strip():
            raise ValueError("ResumeParser: resume_text is empty.")

        message = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[SAVE_RESUME_TOOL],
            tool_choice={"type": "tool", "name": "save_resume"},
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Resume text follows. Call save_resume.\n\n"
                        f"---\n{resume_text}\n---"
                    ),
                }
            ],
        )

        for block in message.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "save_resume":
                data = dict(block.input)
                # Defensive: enforce list defaults so callers never see KeyError.
                data.setdefault("work_experience", [])
                data.setdefault("projects", [])
                data.setdefault("education", [])
                data.setdefault("skills", [])
                return data

        raise RuntimeError(
            "ResumeParser: model did not return a save_resume tool_use block. "
            f"stop_reason={message.stop_reason!r}"
        )
