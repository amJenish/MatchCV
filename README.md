# MatchCV

Resume-driven remote job discovery. Upload your resume, get a curated carousel of legitimate, high-quality remote jobs that actually match your background — no keyword spam, no scams, no in-person roles dressed up as remote.

---

## How it works

1. **Sign in** with Gmail (Supabase OAuth).
2. **Upload your resume** (PDF or DOCX).
3. MatchCV parses your resume into structured sections, derives a *signal profile* (suitable seniority, primary roles, core skills, search keywords, deal-breakers), and immediately starts streaming jobs from five sources.
4. Each candidate runs through a multi-stage pipeline:

   ```
   scrape → prefilter → extract → legitimacy check → quality check → fit check
   ```

   Only jobs clearing all three display gates show up in your carousel:

   - `legitimacy_score > 74`
   - `quality_score    > 79`
   - `fit_score        > 50`

5. Apply, save, or dismiss in the carousel. Your queue restores exactly where you left off.

---

## Tech stack

| Layer | Tools |
|---|---|
| **Backend** | Python · FastAPI · Uvicorn · `asyncio` queues |
| **LLM** | Anthropic Claude Haiku (tool-use) for resume parsing, signal extraction, job extraction, and fit scoring |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| **Database** | Supabase Postgres + `pgvector` |
| **Auth** | Supabase Auth · Google (Gmail) OAuth |
| **Resume parsing** | PyMuPDF (PDF) · python-docx (DOCX) |
| **HTTP / scraping** | httpx · BeautifulSoup4 |
| **Frontend** | Vanilla HTML / CSS / JavaScript (no framework, no build step) |
| **Job sources** | RemoteOK · Remotive · Arbeitnow · WeWorkRemotely · Himalayas |

---

## Repository layout

```
.
├── backend/
│   ├── api/                     # FastAPI routers (auth, resume, jobs, queue)
│   ├── services/                # ResumeService, JobDiscoveryService, ScrapingService
│   ├── parsing/                 # scraper, prefilter, extractor, resume parsers
│   ├── validating/              # legitimacy_check, quality_check, job_fit_checker
│   ├── database/
│   │   ├── database_manager/    # Supabase repositories
│   │   └── *.sql                # schema + RPC migrations
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── index.html               # carousel + main flow
    ├── login.html               # Gmail OAuth entry
    ├── profile.html             # resume re-upload + signal preview
    └── auth.js                  # shared API helpers
```

---

## Setup

### 1. Prerequisites

- Python 3.10+
- A Supabase project (free tier works)
- An Anthropic API key

### 2. Database

In Supabase SQL editor, run the migrations in `backend/database/`:

```sql
-- in order
\i backend/database/queue_schema.sql
\i backend/database/resume_schema.sql
\i backend/database/discovery_rpc.sql
```

Required tables: `profiles`, `scrapelist`, `user_job_interactions`, `work_experience`, `projects`, `education`, `skills`. The `pgvector` extension and a `vector(1536)` column on `scrapelist.job_embedding` are also required.

### 3. Auth

In your Supabase dashboard, enable **Google** as an OAuth provider and set the redirect URI to:

```
http://127.0.0.1:8000/auth/callback
```

### 4. Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS/Linux

pip install -r requirements.txt

cp .env.example .env             # then fill in real values
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Frontend

The frontend is plain static files — open `frontend/login.html` directly, or serve the folder with any static server pointed at the same origin as the API.

---

## Environment variables

See [`backend/.env.example`](backend/.env.example) for the full list. The required variables are:

- `ANTHROPIC_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_KEY` (anon)
- `SUPABASE_SECRET_KEY` (service role)
- `SUPABASE_REDIRECT_URI`

---

## Architecture notes

- **Streaming pipeline.** `ScrapingService` runs scrape → prefilter → verify as three concurrent `asyncio` tasks connected by queues. The carousel populates as jobs are verified, not at the end of a batch.
- **Cheap stages first.** Embedding pre-filter → legitimacy → quality → fit. The most expensive call (LLM fit check) only runs on jobs that already cleared every cheaper gate.
- **Idempotency lock.** `profiles.scraping_in_progress` prevents concurrent discovery runs for the same user.
- **Tool-use only.** Every LLM output is schema-validated through Anthropic tool-use — no free-form JSON parsing.
- **Keyword relaxation ladder.** Strict keywords first, then top-3, then a broad firehose, gated on a minimum candidate count so niche resumes still get a healthy pool.

---

## License

This project is provided as-is for educational and hackathon purposes.
