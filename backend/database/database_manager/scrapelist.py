import logging

from database.database_manager._client import get_client
from parsing.scraper import normalize_url

logger = logging.getLogger(__name__)


def normalize_posting_url(url: str) -> str:
    """Compatibility wrapper: defers to the canonical scraper.normalize_url."""
    return normalize_url(url or "")


def job_to_scrapelist_row(job: dict) -> dict:
    """Map a verified pipeline job dict to a scrapelist table row."""
    posting_url = normalize_posting_url(job.get("posting_url") or "")
    return {
        "posting_url": posting_url,
        "apply_url": job.get("apply_url"),
        "apply_url_source": job.get("apply_url_source"),
        "content_hash": job.get("content_hash"),
        "source": job.get("source"),
        "job_title": job.get("job_title"),
        "company_name": job.get("company_name"),
        "company_domain": job.get("company_domain"),
        "company_domain_extracted": job.get("company_domain_extracted"),
        "company_domain_clearbit": job.get("company_domain_clearbit"),
        "company_logo": job.get("company_logo"),
        "posted_by": job.get("posted_by"),
        "contact_email": job.get("contact_email"),
        "job_description": job.get("job_description"),
        "raw_text": job.get("raw_text"),
        "responsibilities": job.get("responsibilities") or [],
        "required_qualifications": job.get("required_qualifications") or [],
        "preferred_qualifications": job.get("preferred_qualifications") or [],
        "benefits": job.get("benefits") or [],
        "seniority_level": job.get("seniority_level"),
        "is_remote": job.get("is_remote"),
        "remote_region": job.get("remote_region"),
        "timezone_requirements": job.get("timezone_requirements"),
        "work_authorization": job.get("work_authorization"),
        "location": job.get("location"),
        "company_hq_country": job.get("company_hq_country"),
        "salary_min": job.get("salary_min"),
        "salary_max": job.get("salary_max"),
        "salary_currency": job.get("salary_currency"),
        "salary_period": job.get("salary_period"),
        "salary_stated": bool(job.get("salary_stated")),
        "equity_offered": bool(job.get("equity_offered")),
        "posting_date": job.get("posting_date"),
        "deadline": job.get("deadline"),
        "extraction_confidence": job.get("extraction_confidence"),
        "is_truncated_or_stub": bool(job.get("is_truncated_or_stub")),
        "legitimacy_status": job.get("legitimacy_status") or job.get("status"),
        "legitimacy_score": job.get("legitimacy_score"),
        "legitimacy_flags": job.get("legitimacy_flags") or [],
        "legitimacy_company_verification": job.get("legitimacy_company_verification"),
        "quality_score": job.get("quality_score"),
        "meets_standard": bool(job.get("meets_standard")),
        "quality_components": job.get("quality_components") or {},
        "quality_penalties": job.get("quality_penalties") or {},
        "quality_summary": job.get("quality_summary") or "",
        "quality_failures": job.get("quality_failures") or [],
        "quality_highlights": job.get("quality_highlights") or [],
        "pipeline_payload": job,
    }


def _format_pgvector(values: list[float] | tuple[float, ...]) -> str:
    """pgvector accepts strings of the form '[v1,v2,...]' over PostgREST."""
    return "[" + ",".join(f"{float(v):.7f}" for v in values) + "]"


def parse_pgvector(value) -> list[float] | None:
    """Parse a pgvector value as returned by Supabase into a list of floats."""
    if value is None:
        return None
    if isinstance(value, list):
        return [float(x) for x in value]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        s = s.strip("[]")
        if not s:
            return None
        try:
            return [float(x) for x in s.split(",")]
        except ValueError:
            return None
    return None


class ScrapelistRepository:
    TABLE = "scrapelist"

    def _table(self):
        return get_client().table(self.TABLE)

    def posting_url_exists(self, posting_url: str) -> bool:
        normalized = normalize_posting_url(posting_url)
        if not normalized:
            return False
        response = (
            self._table()
            .select("posting_url")
            .eq("posting_url", normalized)
            .limit(1)
            .execute()
        )
        return bool(response.data)

    def lookup_id_by_posting_url(self, posting_url: str | None) -> str | None:
        """Return scrapelist.id for an already-persisted posting_url, or None."""
        normalized = normalize_posting_url(posting_url or "")
        if not normalized:
            return None
        response = (
            self._table()
            .select("id")
            .eq("posting_url", normalized)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        if not rows:
            return None
        return rows[0].get("id")

    def content_hash_exists(self, content_hash: str | None) -> bool:
        if not content_hash:
            return False
        response = (
            self._table()
            .select("content_hash")
            .eq("content_hash", content_hash)
            .limit(1)
            .execute()
        )
        return bool(response.data)

    def insert_verified_job(self, job: dict) -> dict | None:
        """
        Insert a verified job into scrapelist. Skips if posting_url is
        missing or already present, or if content_hash already exists.
        Does not update or delete existing rows.
        """
        row = job_to_scrapelist_row(job)
        posting_url = row.get("posting_url")
        content_hash = row.get("content_hash")

        if not posting_url:
            logger.warning("insert_verified_job: missing posting_url — skipped")
            return None
        if not content_hash:
            logger.warning(
                "insert_verified_job: missing content_hash for %s — skipped",
                posting_url,
            )
            return None

        if self.posting_url_exists(posting_url):
            logger.info("insert_verified_job: posting_url exists — %s", posting_url)
            return None
        if self.content_hash_exists(content_hash):
            logger.info(
                "insert_verified_job: content_hash exists — %s",
                content_hash,
            )
            return None

        try:
            response = self._table().insert(row).execute()
        except Exception:
            logger.exception(
                "insert_verified_job failed — check columns match "
                "database/scrapelist_schema.sql"
            )
            raise

        if not response.data:
            return None
        return response.data[0]

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    def fetch_unseen_for_user_signal(
        self,
        user_id: str,
        *,
        seniority_levels: list[str] | None = None,
        min_legitimacy: int = 50,
        min_quality: int = 35,
        limit: int = 500,
    ) -> list[dict]:
        """
        Fetch scrapelist rows excluded via NOT EXISTS against
        user_job_interactions (any status) using the
        public.scrapelist_unseen_for_user RPC. Single round-trip.

        seniority_levels is a list of scrapelist.seniority_level enum values
        (e.g. ['junior','mid']) used as a pre-filter. Pass None to disable.
        """
        client = get_client()
        try:
            response = client.rpc(
                "scrapelist_unseen_for_user",
                {
                    "p_user_id":          user_id,
                    "p_min_legitimacy":   min_legitimacy,
                    "p_min_quality":      min_quality,
                    "p_seniority_levels": seniority_levels,
                    "p_limit":            limit,
                },
            ).execute()
        except Exception:
            logger.exception(
                "scrapelist_unseen_for_user RPC failed — falling back to "
                "two-query path. Did you apply database/discovery_rpc.sql?"
            )
            return self.fetch_candidates_for_user(
                user_id,
                min_legitimacy=min_legitimacy,
                min_quality=min_quality,
                limit=limit,
            )

        rows = response.data or []
        for row in rows:
            row["job_embedding"] = parse_pgvector(row.get("job_embedding"))
        return rows

    def fetch_candidates_for_user(
        self,
        user_id: str | None,
        *,
        min_legitimacy: int = 50,
        min_quality: int = 35,
        limit: int = 500,
    ) -> list[dict]:
        """
        Return recent scrapelist rows passing the quality/legitimacy gates,
        excluding rows the given user has already interacted with. Each row
        has its `job_embedding` column parsed into a list of floats (or None).
        """
        client = get_client()

        seen_ids: list[str] = []
        if user_id:
            seen_response = (
                client.table("user_job_interactions")
                .select("job_id")
                .eq("user_id", user_id)
                .execute()
            )
            seen_ids = [row["job_id"] for row in (seen_response.data or [])]

        query = (
            client.table(self.TABLE)
            .select("*")
            .gte("legitimacy_score", min_legitimacy)
            .gte("quality_score", min_quality)
            .order("scraped_at", desc=True)
            .limit(limit)
        )
        if seen_ids:
            query = query.not_.in_("id", seen_ids)

        rows = query.execute().data or []
        for row in rows:
            row["job_embedding"] = parse_pgvector(row.get("job_embedding"))
        return rows

    def update_job_embedding(
        self,
        scrapelist_id: str | None,
        embedding: list[float] | None,
    ) -> None:
        """Persist a single job embedding back to the row."""
        if not scrapelist_id or not embedding:
            return
        try:
            self._table().update(
                {"job_embedding": _format_pgvector(embedding)}
            ).eq("id", scrapelist_id).execute()
        except Exception:
            logger.warning(
                "update_job_embedding failed for id=%s",
                scrapelist_id,
                exc_info=True,
            )

    def bulk_update_job_embeddings(
        self,
        items: list[tuple[str, list[float]]],
    ) -> int:
        """Persist multiple embeddings sequentially. Returns count of rows updated."""
        updated = 0
        for scrapelist_id, embedding in items:
            try:
                self.update_job_embedding(scrapelist_id, embedding)
                updated += 1
            except Exception:
                logger.warning(
                    "bulk_update_job_embeddings: failed id=%s",
                    scrapelist_id,
                    exc_info=True,
                )
        return updated
