import anthropic
import json
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)



# Scoring rubric ----------------------------------------------------------------------------

RUBRIC = """
You are a professional technical recruiter scoring candidate fit against a job posting.
Score the candidate using EXACTLY this rubric. Every criterion has fixed point values —
pick the value that best describes the match. Do not interpolate between values.

─────────────────────────────────────────────────────────────
CRITERION 1 — Job title / seniority match                (0–25 pts)
─────────────────────────────────────────────────────────────
  25 pts  Candidate's most recent title is identical or an accepted equivalent
          (e.g. "Software Engineer" ↔ "Software Developer").
  15 pts  Related title in the same domain but different seniority or speciality
          (e.g. "Backend Engineer" for a "Full-Stack Engineer" role).
   5 pts  Adjacent domain — plausible career transition but not a direct match.
   0 pts  Unrelated title with no evident overlap.

─────────────────────────────────────────────────────────────
CRITERION 2 — Required qualifications coverage           (0–35 pts)
─────────────────────────────────────────────────────────────
  Count the total number of required qualifications listed in the job.
  Count how many are explicitly covered by the candidate's profile
  (consider synonyms and equivalent phrasing as matches).
  Award: round((matched / total) * 35) points.
  If the job lists no required qualifications, award 25 pts by default.

─────────────────────────────────────────────────────────────
CRITERION 3 — Relevant work experience depth             (0–25 pts)
─────────────────────────────────────────────────────────────
  25 pts  3 or more years of experience directly relevant to this role.
  15 pts  1–3 years of directly relevant experience.
   5 pts  Experience is adjacent but not directly relevant.
   0 pts  No relevant work experience present.

─────────────────────────────────────────────────────────────
CRITERION 4 — Project relevance                          (0–10 pts)
─────────────────────────────────────────────────────────────
  10 pts  One or more projects directly demonstrate skills required by this job.
   5 pts  Projects are partially relevant — some skill overlap.
   0 pts  No relevant projects, or no projects listed.

─────────────────────────────────────────────────────────────
CRITERION 5 — Education relevance                        (0–5 pts)
─────────────────────────────────────────────────────────────
   5 pts  Degree or certification directly relevant to the role.
   2 pts  Tangentially relevant field of study or certification.
   0 pts  Unrelated education, or no education listed.

─────────────────────────────────────────────────────────────
TOTAL: sum of all five criteria (0–100)
─────────────────────────────────────────────────────────────

Rules:
- Use ONLY the fixed point values listed above. Never use values in between.
- Base your assessment solely on what is explicitly stated in the profile.
  Do not infer or assume skills that are not mentioned.
- Synonyms and equivalent phrasings count as matches for criterion 2.
- Return ONLY valid JSON. No explanation, no markdown fences.
"""

RESPONSE_SCHEMA = """{
    "title_match_score":       <int: 0|5|15|25>,
    "qualifications_score":    <int: 0–35>,
    "experience_score":        <int: 0|5|15|25>,
    "projects_score":          <int: 0|5|10>,
    "education_score":         <int: 0|2|5>,
    "total_score":             <int: 0–100>,
    "matched_requirements":    [<str>, ...],
    "missing_requirements":    [<str>, ...],
    "reasoning":               "<one sentence summary>"
}"""


# Helpers ----------------------------------------------------------------------------

def _format_experience(work_experience: list[dict]) -> str:
    if not work_experience:
        return "None listed."
    lines = []
    for exp in work_experience:
        start   = exp.get("work_start") or "?"
        end     = "Present" if exp.get("is_current") else (exp.get("work_end") or "?")
        bullets = "\n    ".join(exp.get("work_info") or [])
        lines.append(
            f"- {exp.get('work_title', '?')} at {exp.get('company', '?')} "
            f"({start} → {end})\n    {bullets}"
        )
    return "\n".join(lines)


def _format_projects(projects: list[dict]) -> str:
    if not projects:
        return "None listed."
    lines = []
    for proj in projects:
        tech     = ", ".join(proj.get("tech_stack") or [])
        bullets  = "\n    ".join(proj.get("project_info") or [])
        lines.append(
            f"- {proj.get('project_title', '?')} [{tech}]\n    {bullets}"
        )
    return "\n".join(lines)


def _format_education(education: list[dict]) -> str:
    if not education:
        return "None listed."
    lines = []
    for edu in education:
        lines.append(
            f"- {edu.get('credential_name') or edu.get('credential_type', '?')} "
            f"from {edu.get('institution', '?')} "
            f"({edu.get('field_of_study') or 'no field stated'})"
        )
    return "\n".join(lines)


def _format_signals(profile: dict) -> str:
    """
    Surface the resume-derived signal_profile fields that don't fit cleanly
    into work / projects / education. Empty sections render as 'None'.
    The rubric still scores against the job, but giving the LLM these
    signals as additional context makes title and skills matching more
    accurate (and lets it honour deal_breakers + years_of_experience,
    neither of which appear in the bullet-list helpers above).
    """
    yoe = profile.get("years_of_experience")
    primary    = list(profile.get("primary_roles") or [])
    secondary  = list(profile.get("secondary_roles") or [])
    skills     = list(profile.get("core_skills") or [])
    domain     = list(profile.get("domain_expertise") or [])
    keywords   = list(profile.get("search_keywords") or [])
    breakers   = list(profile.get("deal_breakers") or [])

    def _join(items: list[str]) -> str:
        return ", ".join(items) if items else "None"

    yoe_str = f"{int(yoe)} years" if isinstance(yoe, int) else "Not stated"

    return (
        f"Years of experience:    {yoe_str}\n"
        f"Primary target roles:   {_join(primary)}\n"
        f"Secondary target roles: {_join(secondary)}\n"
        f"Core skills:            {_join(skills)}\n"
        f"Domain expertise:       {_join(domain)}\n"
        f"Search keywords:        {_join(keywords)}\n"
        f"Deal-breakers:          {_join(breakers)}"
    )


def _format_remote_policy(job: dict) -> str:
    policy = job.get("is_remote")
    if policy is True:
        return "Remote (true)"
    if policy is False:
        return "Not remote (false)"
    if policy is None:
        return "Not stated"
    return str(policy)


def _format_job(job: dict) -> str:
    responsibilities = "\n  ".join(job.get("responsibilities") or [])
    required         = "\n  ".join(job.get("required_qualifications") or [])
    preferred        = "\n  ".join(job.get("preferred_qualifications") or [])

    return f"""
Job Title:               {job.get("job_title") or job.get("title", "?")}
Company:                 {job.get("company_name") or job.get("company", "?")}
Seniority:               {job.get("seniority_level") or "Not stated"}
Remote Policy:           {_format_remote_policy(job)}

Description:
{job.get("job_description") or job.get("description") or "Not provided."}

Responsibilities:
  {responsibilities or "Not listed."}

Required Qualifications:
  {required or "Not listed."}

Preferred Qualifications:
  {preferred or "Not listed."}
""".strip()


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def _clamp_score(result: dict) -> dict:
    """
    Enforce that individual scores use only allowed values and that
    total_score equals the sum of sub-scores. Guards against any
    off-rubric output despite temperature=0.
    """
    allowed_title   = {0, 5, 15, 25}
    allowed_exp     = {0, 5, 15, 25}
    allowed_proj    = {0, 5, 10}
    allowed_edu     = {0, 2, 5}

    def nearest(value: int, allowed: set[int]) -> int:
        return min(allowed, key=lambda x: abs(x - value))

    title   = nearest(int(result.get("title_match_score",    0)), allowed_title)
    quals   = max(0, min(35, int(result.get("qualifications_score", 0))))
    exp     = nearest(int(result.get("experience_score",     0)), allowed_exp)
    proj    = nearest(int(result.get("projects_score",       0)), allowed_proj)
    edu     = nearest(int(result.get("education_score",      0)), allowed_edu)
    total   = title + quals + exp + proj + edu

    result["title_match_score"]    = title
    result["qualifications_score"] = quals
    result["experience_score"]     = exp
    result["projects_score"]       = proj
    result["education_score"]      = edu
    result["total_score"]          = total   # always recomputed, never trusted from LLM

    return result


# Main class -----------------------------------------------------------

class JobFitChecker:
    """
    Scores how well a candidate profile matches a job posting.

    Returns a score out of 100 using a fixed rubric evaluated by
    Claude Haiku at temperature=0 for deterministic output.

    Usage
    -----
        checker = JobFitChecker()
        result  = checker.check(job=job_dict, profile=profile_dict)
        score   = result["total_score"]   # int 0–100
    """

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model  = "claude-haiku-4-5-20251001"

    def check(self, job: dict, profile: dict) -> dict:
        """
        Parameters
        ----------
        job : dict
            A scrapelist row or extractor output dict.
        profile : dict
            Must contain keys: work_experience, projects, education.
            Each is a list of dicts as returned by the database repositories.

        Returns
        -------
        dict with keys:
            total_score            int  0–100
            title_match_score      int
            qualifications_score   int
            experience_score       int
            projects_score         int
            education_score        int
            matched_requirements   list[str]
            missing_requirements   list[str]
            reasoning              str
        """
        prompt = self._build_prompt(job, profile)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=600,
                temperature=0,          # deterministic
                system=RUBRIC,
                messages=[{"role": "user", "content": prompt}],
            )
            raw    = response.content[0].text
            result = _parse_json(raw)
            result = _clamp_score(result)

        except json.JSONDecodeError as exc:
            logger.error("JobFitChecker: failed to parse LLM response — %s", exc)
            result = self._zero_result("JSON parse error.")

        except Exception as exc:
            logger.error("JobFitChecker: unexpected error — %s", exc)
            result = self._zero_result(str(exc))

        logger.debug(
            "Fit score %d/100 for '%s' @ '%s'",
            result["total_score"],
            job.get("job_title") or job.get("title", "?"),
            job.get("company_name") or job.get("company", "?"),
        )

        return result

    # Internal helpers --------------------------------------------------------------

    def _build_prompt(self, job: dict, profile: dict) -> str:
        return f"""
Score this candidate against the job posting using the rubric provided.

If the candidate's deal-breakers are present in the job posting (e.g. the
job explicitly requires relocation when the candidate listed
"requires relocation" as a deal-breaker), award 0 for the title-match
criterion regardless of title overlap. Use years_of_experience as a
tie-breaker when the work-experience section is sparse but the role's
required experience range is plausible.

════════════════════════════════════════
JOB POSTING
════════════════════════════════════════
{_format_job(job)}

════════════════════════════════════════
CANDIDATE PROFILE
════════════════════════════════════════

CANDIDATE SIGNALS
{_format_signals(profile)}

WORK EXPERIENCE
{_format_experience(profile.get("work_experience", []))}

PROJECTS
{_format_projects(profile.get("projects", []))}

EDUCATION
{_format_education(profile.get("education", []))}

════════════════════════════════════════
Return your assessment as this JSON object (no extra text):
{RESPONSE_SCHEMA}
""".strip()

    @staticmethod
    def _zero_result(reason: str) -> dict:
        return {
            "title_match_score":    0,
            "qualifications_score": 0,
            "experience_score":     0,
            "projects_score":       0,
            "education_score":      0,
            "total_score":          0,
            "matched_requirements": [],
            "missing_requirements": [],
            "reasoning": reason,
        }