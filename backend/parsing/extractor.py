import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


SAVE_JOB_TOOL = {
    "name": "save_job",
    "description": "Save the structured fields of a single job posting.",
    "input_schema": {
        "type": "object",
        "required": [
            "job_title",
            "company_name",
            "job_description",
            "is_remote",
            "extraction_confidence",
            "is_truncated_or_stub",
        ],
        "properties": {
            "job_title": {"type": ["string", "null"]},
            "company_name": {"type": ["string", "null"]},
            "company_domain": {"type": ["string", "null"]},
            "company_logo": {"type": ["string", "null"]},
            "posted_by": {"type": ["string", "null"]},
            "contact_email": {"type": ["string", "null"]},
            "job_description": {"type": ["string", "null"]},
            "responsibilities": {"type": "array", "items": {"type": "string"}},
            "required_qualifications": {"type": "array", "items": {"type": "string"}},
            "preferred_qualifications": {"type": "array", "items": {"type": "string"}},
            "benefits": {"type": "array", "items": {"type": "string"}},
            "seniority_level": {
                "type": ["string", "null"],
                "enum": [
                    "intern", "junior", "mid", "senior",
                    "staff", "principal", "manager", None,
                ],
            },
            "is_remote": {"type": ["boolean", "null"]},
            "remote_region": {"type": ["string", "null"]},
            "timezone_requirements": {"type": ["string", "null"]},
            "work_authorization": {"type": ["string", "null"]},
            "location": {"type": ["string", "null"]},
            "company_hq_country": {"type": ["string", "null"]},
            "salary_min": {"type": ["string", "null"]},
            "salary_max": {"type": ["string", "null"]},
            "salary_currency": {"type": ["string", "null"]},
            "salary_period": {
                "type": ["string", "null"],
                "enum": ["hourly", "weekly", "monthly", "yearly", "contract", None],
            },
            "salary_stated": {"type": "boolean"},
            "equity_offered": {"type": "boolean"},
            "posting_date": {"type": ["string", "null"]},
            "deadline": {"type": ["string", "null"]},
            "apply_url": {"type": ["string", "null"]},
            "extraction_confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "is_truncated_or_stub": {"type": "boolean"},
        },
    },
}


SYSTEM_PROMPT = """You are an extraction tool for remote job postings. Output strictly via the save_job tool. Never write prose.

EXTRACTION RULES
1. Return null when a field is not explicitly present in the posting. Do not infer, guess, or fabricate.
2. job_title: take the literal title from the posting. If only present in the URL slug, you may use that.
3. company_name: prefer the in-text name. If the in-text name is missing but the URL slug or canonical domain unambiguously contains it (e.g. slug "...-aisle-and-abroad-..."), set it from there. If still ambiguous, return null.
4. is_remote: true only if the posting explicitly states remote / distributed / work-from-home / anywhere / etc. False if it explicitly requires on-site presence. null otherwise. Do NOT default to false.
5. salary_min / salary_max: digits-only strings (e.g. "80000"). If only a single salary is stated, set both to the same value. Use salary_period for hourly/weekly/monthly/yearly/contract and salary_currency for the currency code. salary_stated must be true iff a numeric figure is present.
6. timezone_requirements: copy the timezone exactly ("EST", "CET +/- 2h", etc.). If the posting explicitly says any timezone, return "any". Otherwise null.
7. responsibilities / required_qualifications / preferred_qualifications / benefits: short bullet strings, max ~200 chars each, max 12 items each. Skip marketing fluff like "be amazing" or "join our journey".
8. is_truncated_or_stub: true if the body is clearly a redirect/preview to another site under ~400 chars, missing responsibilities AND requirements, or shows aggregator artifacts ("See this and similar jobs on LinkedIn", "Anunciada hh:mm:ss", "Posted hh:mm:ss"). When true, set extraction_confidence <= 0.2.
9. extraction_confidence: a self-rating in [0,1] for how complete and parseable the posting was. Set <= 0.4 if responsibilities and required_qualifications are both empty.
10. Ignore RemoteOK anti-spam suffixes that say "Please mention the word ..." or "tag ROjox" - these are platform artifacts, not job content.
11. Ignore generic AI-tools-in-hiring boilerplate.
12. apply_url: only when an explicit application link appears in the posting (e.g. "Apply here: https://..."). Do not invent URLs from company names. Return null if only a board listing URL is shown.

ANTI-FABRICATION
- It is correct and required to return null. A null field is better than a guessed one.
- If the posting is in a non-English language, extract literally; do not translate.
"""


class JobExtractor:

    def __init__(self) -> None:
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = "claude-haiku-4-5-20251001"

    def extract(self, raw_text: str, posting_url: str | None = None) -> dict:
        print(f"[Extractor] Calling Claude Haiku — url={posting_url!r}", flush=True)

        message = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=[SAVE_JOB_TOOL],
            tool_choice={"type": "tool", "name": "save_job"},
            messages=[{
                "role": "user",
                "content": (
                    f"Posting URL: {posting_url or '(unknown)'}\n\n"
                    f"Posting body:\n{raw_text}"
                ),
            }],
        )

        data = self._extract_tool_input(message)
        data["raw_text"] = raw_text
        data["posting_url"] = posting_url
        print(
            f"[Extractor] Done — job_title={data.get('job_title')!r}, "
            f"company_name={data.get('company_name')!r}, "
            f"confidence={data.get('extraction_confidence')!r}, "
            f"is_truncated_or_stub={data.get('is_truncated_or_stub')!r}",
            flush=True,
        )
        return data

    @staticmethod
    def _extract_tool_input(message) -> dict:
        for block in message.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "save_job":
                return dict(block.input)
        raise RuntimeError(
            "Extractor: model did not return a save_job tool_use block. "
            f"stop_reason={message.stop_reason!r}"
        )
