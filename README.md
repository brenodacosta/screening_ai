# screening_ai

Automated tool to screen candidates for a delivery company.

A two-service prototype: a Chainlit chat UI talks to a FastAPI backend that runs a
LangGraph screening agent against a job description loaded from markdown.

- **Process design:** [docs/process_design.md](docs/process_design.md)
- **Architecture:** [docs/architecture.md](docs/architecture.md)

## Layout

```
screening_ai/
├── api/                    FastAPI app + LangGraph agent
│   ├── main.py             HTTP endpoints + lifespan (saver, sweeper)
│   ├── graph.py            LangGraph state graph compilation
│   ├── nodes.py            extract / route / render / terminate
│   ├── models.py           Pydantic state, candidate, turn, stage
│   ├── prompts.py          extraction + render system prompts
│   ├── llm.py              Copilot OpenAI-compatible client
│   ├── jd_loader.py        YAML-frontmatter JD parser
│   ├── storage.py          candidate JSON, logs, notifications
│   ├── inactivity.py       background sweeper for abandoned chats
│   ├── config.py           pydantic-settings
│   └── Dockerfile
├── ui/                     Chainlit frontend
│   ├── chainlit_app.py     thin client → FastAPI
│   ├── chainlit.md         welcome page
│   └── Dockerfile
├── jobs/
│   └── delivery_driver.md  JD (frontmatter + body) — 45 service areas
├── tests/                  pytest scaffolds (jd, models, routing)
├── docs/                   process_design.md, architecture.md
├── data/                   runtime data (gitignored): checkpoints.db, candidates/, logs/
├── docker-compose.yml
├── requirements.txt
├── .env / .env.example
└── .gitignore
```

## Setup

```powershell
cp .env.example .env       # then edit .env and set COPILOT_PAT
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run (local, two terminals)

```powershell
# Terminal 1 — API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — UI
chainlit run ui/chainlit_app.py --host 0.0.0.0 --port 8001
```

Open <http://localhost:8001> in your browser.

## Run (docker compose)

```powershell
docker compose up --build
```

UI on <http://localhost:8001>, API on <http://localhost:8000>.

## Test

```powershell
pytest -q
```

The included tests are deterministic — they cover JD parsing, model defaults, and
routing logic. LLM-driven scenario tests will be added next.

## What the agent does

1. Greets the candidate bilingually (ES + EN one-liner), then locks to the language
   they reply in. Switches language back and forth turn-by-turn as the candidate does.
2. Walks through 7 questions: name → license → city → availability → preferred
   schedule → prior experience → start date.
3. Hard-disqualifies on: no driver's license, city outside JD service areas, or
   no availability (full-time / part-time / weekends all denied).
4. Writes a JSON record per conversation to `data/candidates/{qualified,disqualified,abandoned}/`
   and appends qualified candidates to `data/notifications.jsonl`.
5. Per-turn JSONL log at `data/logs/{conversation_id}.jsonl`.

## Configuration

All knobs live in `.env`. See [.env.example](.env.example) for documentation.

## LLM backend

We use the **GitHub Copilot SDK** (`github_copilot_sdk`), wrapped as a LangChain
`BaseChatModel` in [api/llm.py](api/llm.py). The SDK spawns the bundled
`copilot` CLI subprocess and handles PAT → session-token exchange internally —
which is what makes a fine-grained GitHub PAT work as the LLM credential.

Default model is `claude-haiku-4.5` (non-premium, fast, cheap). Switch via
`MODEL_NAME` in `.env`. See [docs/architecture.md §4.6](docs/architecture.md#46-llm-client-copilot-sdk-wrapper) for the full design rationale,
including why we run the SDK on its own worker-thread asyncio loop and what
quirks to expect from the CLI's prompt injection.

## Smoke test

```powershell
.venv\Scripts\python.exe smoke_test.py
```

Should print `OK : received '...'` — your PAT is wired up and the SDK can
reach the Copilot backend. Run this first if anything looks off; it bypasses
LangGraph and FastAPI to test just the LLM client.
