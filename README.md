# screening_ai

Automated tool to screen delivery-driver candidates via chat. A Chainlit chat UI
talks to a FastAPI backend that runs a LangGraph screening agent against a job
description loaded from markdown.

- **Process design:** [docs/process_design.md](docs/process_design.md)
- **Architecture:** [docs/architecture.md](docs/architecture.md)

## What the agent does

1. **Bilingual greeting** (ES + EN one-liner), then locks to the language the
   candidate replies in. Switches back and forth turn-by-turn freely.
2. **Walks through 7 questions:** name → license → city → availability →
   preferred schedule → prior delivery experience → start date.
3. **Answers candidate questions from the JD** (pay, perks, requirements, etc.)
   while staying on-task. Strictly grounded in [jobs/delivery_driver.md](jobs/delivery_driver.md);
   prompt-injection-resistant — *"ignore previous instructions"* style messages
   get a polite refusal, not a leak.
4. **Hard-disqualifies** on: no driver's license, city outside JD service
   areas, or no availability at all.
5. **Inactivity handling:** one nudge ("¿sigues ahí?" + current question), then
   close as `abandoned` if the candidate still doesn't reply. Triggered by a
   real 10-min sweeper *and* a **User inactivity** button in Chainlit for
   simulation.
6. **Writes a JSON record** per conversation to
   `data/candidates/{qualified,disqualified,abandoned}/` and appends qualified
   candidates to `data/notifications.jsonl`.
7. **Per-turn log** at `data/logs/{conversation_id}.jsonl` (extractions, render
   calls, Q&A calls, fallbacks, inactivity events).
8. **Source flag on every agent message** — Chainlit shows the author as
   **Assistant** (LLM-generated) or **Template** (deterministic fallback /
   greeting / closing) so you can see at a glance what the model did vs. what
   the code did.

## Layout

```
screening_ai/
├── api/                    FastAPI app + LangGraph agent
│   ├── main.py             HTTP endpoints + lifespan (saver, sweeper)
│   ├── graph.py            LangGraph state graph compilation
│   ├── nodes.py            extract / route / render / terminate / Q&A
│   ├── models.py           Pydantic state, candidate, turn (with source flag)
│   ├── prompts.py          extraction / render / QA system prompts + templates
│   ├── llm.py              langchain-openai ChatOpenAI (provider-agnostic)
│   ├── jd_loader.py        YAML-frontmatter JD parser
│   ├── storage.py          candidate JSON, logs, notifications
│   ├── inactivity.py       background sweeper (nudge → abandon)
│   ├── config.py           pydantic-settings
│   └── Dockerfile
├── ui/                     Chainlit frontend
│   ├── chainlit_app.py     thin client → FastAPI + inactivity button
│   ├── chainlit.md         welcome page
│   └── Dockerfile
├── jobs/
│   └── delivery_driver.md  JD (frontmatter + body) — 45 service areas
├── tests/                  pytest scaffolds (JD loader, models, routing)
├── docs/                   process_design.md, architecture.md
├── data/                   runtime data (gitignored): checkpoints.db, candidates/, logs/
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Prerequisites

- **Python 3.12.x** (3.13 is **not** supported — chainlit's transitive deps
  pull `numpy<2` which has no 3.13 wheel). On Windows: `winget install
  Python.Python.3.12`. On Linux: see the platform-specific steps below.
- **git**, of course.
- **Free Google Gemini API key** — get one instantly at
  <https://aistudio.google.com/apikey>. Free tier covers prototype usage
  (~1,500 requests/day on `gemini-2.0-flash`).
- Optional: Docker + Docker Compose, if you want the containerised path.

---

## Quick start — Windows (PowerShell)

```powershell
# 1. Get the code
git clone git@github.com:brenodacosta/screening_ai.git
cd screening_ai

# 2. Configure
Copy-Item .env.example .env
# Open .env and paste your Gemini key on the LLM_API_KEY= line.

# 3. Virtual env (Python 3.12)
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Smoke test the LLM credential (optional but recommended)
python smoke_test.py
# expected: "OK : received '...'"

# 5. Run
# -- Terminal 1 (API)
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# -- Terminal 2 (UI)
.venv\Scripts\python.exe -m chainlit run ui\chainlit_app.py --host 127.0.0.1 --port 8001 --headless
```

Open <http://127.0.0.1:8001> in your browser.

---

## Quick start — Linux / macOS (bash / zsh)

```bash
# 1. Get the code
git clone git@github.com:brenodacosta/screening_ai.git
cd screening_ai

# 2. Configure
cp .env.example .env
# Open .env and paste your Gemini key on the LLM_API_KEY= line:
#   nano .env   (or your editor of choice)

# 3. Virtual env (Python 3.12)
#    Ubuntu/Debian:  sudo apt install python3.12 python3.12-venv
#    macOS (brew) :  brew install python@3.12
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Smoke test the LLM credential (optional but recommended)
python smoke_test.py
# expected: "OK : received '...'"

# 5. Run
# -- Terminal 1 (API)
.venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# -- Terminal 2 (UI)
.venv/bin/python -m chainlit run ui/chainlit_app.py --host 127.0.0.1 --port 8001 --headless
```

Open <http://127.0.0.1:8001> in your browser.

---

## Run via Docker Compose (Windows / Linux / macOS)

Requires Docker Desktop (Win/macOS) or Docker Engine + Compose plugin (Linux).
`.env` is read by the `api` container via `env_file:`; make sure
`LLM_API_KEY` is populated before bringing up.

```bash
docker compose up --build
```

- UI: <http://localhost:8001>
- API: <http://localhost:8000> (healthcheck on `/healthz`)

Stop with `docker compose down`. Persistent state lives in the named volume
`screening_data` (checkpoints + candidate JSONs + logs).

---

## Tests

```bash
pytest -q
```

Deterministic tests covering JD parsing, model defaults, and routing. LLM-driven
scenario tests are tracked in [docs/architecture.md §8](docs/architecture.md#8-observability--testing).

---

## Configuration

All knobs live in `.env`. See [.env.example](.env.example) for the full
reference. The three you'll likely touch:

| Variable | Default | Notes |
|---|---|---|
| `LLM_API_KEY` | (empty) | Google Gemini key, or any OpenAI-compatible provider key |
| `LLM_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` | Swap to GitHub Models, Groq, Ollama, etc. |
| `MODEL_NAME` | `gemini-2.5-flash` | Model id for whatever endpoint `LLM_BASE_URL` points at |

`INACTIVITY_TIMEOUT_SECONDS` controls the real 10-min nudge/abandon sweeper.
Lower it (e.g. `30`) to test the inactivity flow without waiting.

---

## Dashboard

Aggregated metrics over every terminal candidate JSON on disk, computed
on-demand per request (no cache, no background recompute — just point the
endpoint at `data/candidates/`):

- `GET /dashboard` — self-contained HTML page with Plotly charts (loads
  Plotly from a public CDN; charts won't render fully offline).
- `GET /dashboard/metrics` — same data as JSON. Both accept an optional
  `?job_id=` filter; omit to aggregate across all jobs.

Metrics: totals per status, completion rate, drop-off by stage,
average duration (overall + per status), disqualification reason
breakdown, 7-stage funnel, LLM-vs-Template message ratio, and daily
throughput. Open <http://127.0.0.1:8000/dashboard> after starting the API.

---

## LLM backend

We use `langchain-openai`'s `ChatOpenAI` wired up in [api/llm.py](api/llm.py)
against an OpenAI-compatible endpoint. Default target: **Google Gemini 2.5
Flash** via Gemini's OpenAI bridge. Three env vars decide everything:
`LLM_API_KEY`, `LLM_BASE_URL`, `MODEL_NAME`. Zero code changes to swap
providers (GitHub Models, Groq, Ollama, OpenRouter, an internal proxy).

See [docs/architecture.md §4.6](docs/architecture.md#46-llm-client-gemini-via-openai-compatible-endpoint)
for the full provider-pivot history (Copilot Chat → Copilot SDK → GitHub Models
direct → Gemini) and the lessons learned.

## Smoke test

```bash
# Windows
.venv\Scripts\python.exe smoke_test.py

# Linux / macOS
.venv/bin/python smoke_test.py
```

Should print `OK : received '...'` — your key works and the provider is
reachable. If it fails, this is the cleanest place to debug; it bypasses
LangGraph, FastAPI, and Chainlit entirely.
