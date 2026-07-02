# SHL Assessment Recommender

Conversational agent that takes a hiring manager from a vague ask to a grounded
shortlist of SHL assessments. FastAPI + Gemini (native `google-genai` SDK, no
framework). Built for the SHL Labs AI Intern take-home
(`SHL_AI_Intern_Assignment.pdf`).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in GEMINI_API_KEY
```

Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
**Before relying on it for anything real (including the eval harness or the
deployed endpoint), check your quota in AI Studio or enable billing** — see
`APPROACH.md`'s "Evaluation" section for why this matters and isn't optional.

## Run locally

```bash
uvicorn app.main:app --reload
```

- `GET http://localhost:8000/health` -> `{"status": "ok"}`
- `POST http://localhost:8000/chat` -> see API spec in the assignment PDF; request/response shapes are in `app/schemas.py`.

## Test

```bash
pytest tests/ -v                 # fast, no network/API key needed -- mocks the LLM call
python scripts/eval_harness.py   # slow, needs a real GEMINI_API_KEY -- replays the 10 gold traces
```

## Project layout

```
app/
  main.py        FastAPI app: /health, /chat
  agent.py        turn logic: prefilter -> generate -> validate (no framework)
  catalog.py       catalog loading, grounding lookups, compact prompt serialization
  guardrails.py     rule-based prompt-injection prefilter
  prompts.py         system prompt + prompt builders
  schemas.py          request/response models
  config.py            env-driven settings
data/
  catalog.json   377 SHL Individual Test Solutions (see scripts/extract_catalog.py for provenance)
scripts/
  extract_catalog.py  reproduces data/catalog.json from the source PDF
  trace_parser.py     parses traces/*.md into structured gold turns
  eval_harness.py      replays gold traces against the live agent, reports recall@10
  manual_check.py       10-second eyeball check against a running server
traces/
  C1.md ... C10.md     the 10 reference conversations provided with the assignment
tests/
  test_smoke.py         plumbing tests with the LLM mocked out
```

## Deploy (Render)

1. New Web Service -> connect this repo -> Docker runtime (uses the included `Dockerfile`).
2. Set env var `GEMINI_API_KEY` in the Render dashboard.
3. Health check path: `/health`.

See `APPROACH.md` for design rationale, including the real provider/framework
history (Gemini+LangGraph -> Groq+BM25 -> Gemini+native SDK) and why each
change happened.
