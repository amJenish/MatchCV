"""
Integration test for ScrapingService.

Runs the full pipeline:
    scrape -> prefilter -> Extract -> stub-gate -> Legitimacy -> legit-gate
        -> Quality -> quality-gate -> persist.

Requirements:
  - backend/.env with SUPABASE_URL, SUPABASE_SECRET_KEY, ANTHROPIC_API_KEY
  - Network access to job boards and APIs (RemoteOK, Remotive,
    WeWorkRemotely, Himalayas, Clearbit, Anthropic)

Run from the backend directory:
  python -m unittest test.test_scraping_service -v

Or:
  python test/test_scraping_service.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import unittest
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

load_dotenv(_BACKEND_ROOT / ".env")

from database.database_manager.scrapelist import ScrapelistRepository
from parsing.constants import QUEUE_END
from services.ScrapingService import MIN_QUALITY_SCORE, ScrapingService

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_FILE = OUTPUT_DIR / "scraping_service_10_jobs.json"

# Allow up to 60 minutes — primary fetches + N×(extract + legit + quality).
RUN_TIMEOUT_SECONDS = 3600

SAMPLE_PROFILE: dict = {}

REQUIRED_JOB_KEYS = {
    "job_title",
    "company_name",
    "posting_url",
    "apply_url",
    "apply_url_source",
    "content_hash",
    "source",
    "status",
    "legitimacy_status",
    "legitimacy_score",
    "legitimacy_flags",
    "legitimacy_company_verification",
    "extraction_confidence",
    "is_truncated_or_stub",
    "quality_score",
    "quality_components",
    "quality_penalties",
    "meets_standard",
}

LEGITIMACY_FLOOR = 50


def _env_ready() -> bool:
    return bool(
        os.getenv("SUPABASE_URL")
        and os.getenv("SUPABASE_SECRET_KEY")
        and os.getenv("ANTHROPIC_API_KEY")
    )


def _json_default(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


async def _run_scraping_service(n: int = 10) -> dict:
    print(f"\n[TEST] Starting ScrapingService integration test (n={n})", flush=True)
    print(
        "[TEST] Primary sources: RemoteOK + Remotive + WeWorkRemotely + Himalayas; "
        "Arbeitnow is secondary fallback\n",
        flush=True,
    )

    result_queue: asyncio.Queue = asyncio.Queue()
    service = ScrapingService(n=n)

    print("[TEST] Calling service.run() — pipeline output will stream below\n", flush=True)
    verified_count = await asyncio.wait_for(
        service.run(SAMPLE_PROFILE, result_queue),
        timeout=RUN_TIMEOUT_SECONDS,
    )
    print(f"\n[TEST] service.run() returned — verified_count={verified_count}", flush=True)

    jobs: list[dict] = []
    print("[TEST] Draining result queue...", flush=True)
    while True:
        item = await result_queue.get()
        if item is QUEUE_END:
            print("[TEST] Received QUEUE_END on result queue", flush=True)
            break
        jobs.append(item)
        print(f"[TEST] Collected job {len(jobs)}: {item.get('job_title')}", flush=True)

    return {
        "requested_prefilter_passes": n,
        "prefilter_passed": service.passed_count,
        "verified_count": verified_count,
        "inserted_count": service.inserted_count,
        "jobs_collected": len(jobs),
        "jobs": jobs,
    }


def _write_output(payload: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return OUTPUT_FILE


@unittest.skipUnless(
    _env_ready(),
    "SUPABASE_URL, SUPABASE_SECRET_KEY, and ANTHROPIC_API_KEY required",
)
class TestScrapingService(unittest.IsolatedAsyncioTestCase):
    async def test_scrape_ten_jobs(self) -> None:
        logging.basicConfig(level=logging.INFO)
        print("[TEST] test_scrape_ten_jobs — env OK, beginning run\n", flush=True)

        payload = await _run_scraping_service(n=10)
        print("[TEST] Writing JSON output...", flush=True)
        out_path = _write_output(payload)
        print(f"[TEST] Wrote {len(payload['jobs'])} jobs to {out_path}", flush=True)

        jobs = payload["jobs"]
        print(
            f"\nPrefilter passed: {payload['prefilter_passed']}/10 | "
            f"Verified: {payload['verified_count']} | "
            f"Inserted to scrapelist: {payload['inserted_count']} | "
            f"Written to: {out_path}\n"
        )

        for i, job in enumerate(jobs, 1):
            print(
                f"  {i}. {job.get('job_title')} @ {job.get('company_name')} "
                f"[{job.get('status')}] "
                f"legitimacy={job.get('legitimacy_score')} "
                f"quality={job.get('quality_score')} "
                f"{job.get('posting_url')}"
            )

        # Counts.
        self.assertEqual(
            payload["verified_count"],
            payload["jobs_collected"],
            "Verified count must match drained result queue length",
        )
        self.assertEqual(
            payload["verified_count"],
            payload["inserted_count"],
            "Every verified job must also be persisted",
        )
        self.assertGreater(payload["prefilter_passed"], 0, "No jobs passed prefilter")
        self.assertGreater(
            payload["inserted_count"],
            0,
            "No rows inserted into scrapelist — check table schema matches "
            "database/scrapelist_schema.sql",
        )

        # Persistence sanity.
        repo = ScrapelistRepository()
        for job in jobs:
            url = job.get("posting_url")
            self.assertTrue(
                repo.posting_url_exists(url),
                f"Expected scrapelist row for {url}",
            )

        # Shape, status enum, and score floors.
        seen_hashes: set[str] = set()
        for job in jobs:
            missing = REQUIRED_JOB_KEYS - job.keys()
            self.assertFalse(
                missing,
                f"Job missing keys {missing}: {job.get('job_title')}",
            )
            self.assertIn(
                job.get("status"),
                ("legitimate", "warning"),
                "Fraud jobs must not appear in results",
            )
            self.assertEqual(
                job.get("status"),
                job.get("legitimacy_status"),
                "status mirror must match legitimacy_status",
            )
            self.assertGreaterEqual(
                job.get("legitimacy_score", -1),
                LEGITIMACY_FLOOR,
                f"legitimacy_score below floor for {job.get('job_title')!r}",
            )
            self.assertLessEqual(job.get("legitimacy_score", 101), 100)
            self.assertGreaterEqual(
                job.get("quality_score", -1),
                MIN_QUALITY_SCORE,
                f"quality_score below floor for {job.get('job_title')!r}",
            )
            self.assertLessEqual(job.get("quality_score", 101), 100)

            # content_hash uniqueness within this run.
            ch = job.get("content_hash")
            self.assertTrue(ch, f"Missing content_hash for {job.get('job_title')!r}")
            self.assertNotIn(
                ch,
                seen_hashes,
                f"Duplicate content_hash within run for {job.get('job_title')!r}",
            )
            seen_hashes.add(ch)


if __name__ == "__main__":
    unittest.main(verbosity=2)
