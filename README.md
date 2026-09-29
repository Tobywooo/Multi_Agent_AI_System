# Group Project 1: Two-Agent AI System (A2A + RAG + Playwright)

A **Requester Agent** takes an IT support request from the user and delegates it
to a **Specialist Agent** over a simple A2A-style HTTP protocol. The Specialist
answers with RAG over the support knowledge base. The Requester then uses
**Playwright** to file a ticket in the mock support app with the Specialist's
category and resolution, and verifies the confirmation.

```
User request -> Requester Agent -> Specialist Agent -> RAG -> Specialist response
             -> Requester Agent -> Playwright -> Verification
```

## Contents

- [Setup](#setup)
- [Running the system](#running-the-system)
- [Demo scenarios](#demo-scenarios)
- [How it works](#how-it-works)
- [A2A protocol](#a2a-protocol)
- [RAG pipeline](#rag-pipeline)
- [Failure handling](#failure-handling)
- [Testing components individually](#testing-components-individually)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## Setup

Requires Python 3.11+ and a free [Groq API key](https://console.groq.com/keys).
Run everything from the project folder (the one containing this README).

**1. Create and activate a virtual environment**

```bash
python -m venv .venv
```

Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:
```bash
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`. Activate it in **every** new
terminal. (Or skip activation and call `.venv/Scripts/python` on Windows /
`.venv/bin/python` on macOS/Linux instead of `python`.)

**2. Install dependencies and the Playwright browser**

```bash
pip install -r requirements.txt
playwright install chromium
```

**3. Add your API key**

Copy `.env.example` to `.env` in the project folder and fill in the key:

```
GROQ_API_KEY=gsk_...
```

`.env` is git-ignored. **Never commit or submit it.**

**4. Build the vector store**

```bash
python -m rag.ingest
```

This embeds the knowledge base into `vector_store/` (a few seconds; the first
run also downloads the embedding model). Re-run it whenever `knowledge_base/`
changes.

## Running the system

Use two terminals, both in the project folder with the venv activated.

**Terminal 1: start the Specialist Agent** (leave it running)

```bash
python -m specialist
```

Wait for `Uvicorn running on http://127.0.0.1:8001`. The first start takes
10-20 seconds while the models load.

**Terminal 2: start the Requester Agent**

```bash
python -m requester
```

This opens an interactive session. Type a support request at the `Request>`
prompt and press Enter. For each request the Requester:

1. sends a task to the Specialist and prints each status change,
2. prints the Specialist's category, resolution, and cited sources,
3. opens a browser, fills in and submits the ticket form, and
4. verifies the confirmation and prints the ticket number.

Enter a blank line or `quit` to end the session.

Single request instead of a session:

```bash
python -m requester "My keyboard has stopped working."
```

| Option | Default | Meaning |
|---|---|---|
| `--timeout` | `30` | Seconds to wait for the Specialist before giving up |
| `--poll-interval` | `1` | Seconds between status checks |
| `--headless` | off | Hide the browser and skip the demo pauses |
| `--slow-mo` | `300` | Milliseconds between browser actions |
| `--hold` | `3` | Seconds to keep the confirmation on screen |
| `--simulate slow\|fail` | none | Single-request mode: make the Specialist time out or fail |
| `--specialist-url` | `http://127.0.0.1:8001` | Where the Specialist is running |

While the Specialist is running you can also open http://127.0.0.1:8001 in a
browser for interactive API docs (submit and poll tasks by hand), or
http://127.0.0.1:8001/.well-known/agent.json for its agent card.

## Demo scenarios

In an interactive session, prefix a request with `/fail` or `/slow` to simulate
a Specialist failure or timeout. Start the session with a shorter timeout
(`python -m requester --timeout 10`) so the timeout demo doesn't take 30 s.

| Type at `Request>` | What happens |
|---|---|
| `My keyboard has stopped working.` | Success: Hardware ticket filed and verified |
| `I forgot my password and cannot log into my account.` | Success: Account Access ticket |
| `The whole office lost internet access.` | Success: Network ticket, notes say escalation recommended |
| `/fail My laptop won't connect to Wi-Fi.` | Specialist task `failed` (`simulated_failure`) |
| `/slow My laptop won't connect to Wi-Fi.` | Requester times out while task is still `working` |
| `How do I bake a chocolate cake?` | Task `failed` (`insufficient_context`), no LLM call made |
| `I think someone hacked my account.` | Classified as Security, which the form doesn't offer; routed manually |
| `My company email is not synchronizing.` | Classified as Email, which the form doesn't offer; routed manually |
| *(stop the Specialist with Ctrl+C, then send any request)* | Specialist unavailable |

## How it works

| Step | Component | File |
|---|---|---|
| 1. Receive the user's request | Requester Agent | `requester/__main__.py` |
| 2. Build the task (question + the form's allowed categories) and submit it | Requester Agent, A2A client | `requester/agent.py`, `a2a_protocol/client.py` |
| 3. Store the task, acknowledge with a task ID, run it in the background | A2A server | `a2a_protocol/server.py` |
| 4. Retrieve, rerank, and generate a grounded answer | Specialist Agent, RAG | `specialist/agent.py`, `rag/` |
| 5. Poll until `completed` / `failed` or the timeout | A2A client | `a2a_protocol/client.py` |
| 6. Check the category exists on the form | Requester Agent | `requester/agent.py` |
| 7. Fill in, submit, and verify the ticket | Playwright | `automation/ticket_form.py` |

The Specialist's answer directly drives the browser: the **Category** dropdown
and **Resolution Notes** field are filled from its response, and the notes
include the cited knowledge-base sources.

The mock support app (`mock_support_app/`) is used unchanged. The Playwright
workflow serves it on a temporary local port for each run.

## A2A protocol

Plain HTTP + JSON, modelled on the A2A task lifecycle. Schemas live in
`a2a_protocol/models.py`.

| Endpoint | Purpose |
|---|---|
| `GET /.well-known/agent.json` | Agent card: name, skills, endpoints |
| `POST /tasks` | Submit a task. Returns **202** with `task_id` and status `submitted` immediately |
| `GET /tasks/{task_id}` | Current status, timestamped status history, and the `result` or `error`. Unknown IDs return **404** |

Task statuses: `submitted` -> `working` -> `completed` or `failed`.

Task request (Requester -> Specialist):

```json
{
  "question": "My keyboard has stopped working.",
  "allowed_categories": ["Account Access", "Hardware", "Software", "Network"],
  "metadata": {}
}
```

Completed task result (Specialist -> Requester, abridged):

```json
{
  "task_id": "435163241de0",
  "status": "completed",
  "history": [
    {"status": "submitted", "timestamp": "2026-09-28T19:02:11.104Z"},
    {"status": "working",   "timestamp": "2026-09-28T19:02:11.105Z"},
    {"status": "completed", "timestamp": "2026-09-28T19:02:12.046Z"}
  ],
  "result": {
    "category": "Hardware",
    "kb_category": "Hardware",
    "resolution": "Confirm the keyboard is properly connected ...",
    "escalate": false,
    "confidence": 0.465,
    "sources": [{"source": "hardware.md", "section": "Symptoms", "chunk_id": "hardware:symptoms:0",
                 "score": 0.465, "rerank_score": -3.552}]
  }
}
```

A failed task has `"status": "failed"` and an `error` such as
`{"code": "insufficient_context", "message": "..."}`.

## RAG pipeline

| Stage | Implementation | File |
|---|---|---|
| Load | 7 markdown docs from `knowledge_base/` | `rag/ingest.py` |
| Chunk | Split on `#` / `##` headers: one chunk per section (34 chunks) | `rag/ingest.py` |
| Enrich | **Contextual chunk headers**: document title, section, ticket category | `rag/ingest.py` |
| Embed | `sentence-transformers/all-MiniLM-L6-v2` (local, normalized) | `rag/vector_store.py` |
| Store | FAISS, inner product = cosine similarity | `rag/vector_store.py` |
| Retrieve | Top 10 by similarity; drop anything below 0.2 | `rag/retriever.py` |
| Rerank | **Cross-encoder** `ms-marco-MiniLM-L-6-v2`; keep top 4 | `rag/retriever.py` |
| Generate | Groq `openai/gpt-oss-20b`, strict JSON schema, passages only | `rag/generator.py` |

Settings are in `rag/config.py`. Set `GROQ_MODEL` in `.env` to use a different
Groq model.

### Advanced techniques

**Contextual Chunk Headers** ([rag_techniques](https://github.com/NirDiamant/rag_techniques) #10).
Every knowledge-base doc uses the same section layout, so a bare "Resolution"
chunk from one doc looks a lot like another doc's. Prepending the document
title, section name, and ticket category gives each embedding that context.
`python -m rag.ingest --compare` measures this: **5/5** test cases retrieve the
right category first with headers, **4/5** without. "My laptop won't connect to
Wi-Fi" goes to the Hardware doc without headers.

**Intelligent Reranking** (#17). A cross-encoder reads the question and each
candidate chunk together and scores their relevance directly. This fixes cases
the vector search gets wrong: "I think someone hacked my account" ranks
`account_access` first by similarity, but `security` first after reranking.
Rerank scores are unbounded logits and don't separate real from off-topic
questions cleanly, so the relevance cutoff uses the similarity score instead.

### Grounding

- The LLM sees only the retrieved passages and is told not to add steps they
  don't mention.
- Its JSON schema is built per request: `kb_category` can only be a category
  that appears in the retrieved passages, and `cited_chunks` can only be their
  IDs.
- Mapping to the form's categories happens in code, not in the LLM. An Email or
  Security issue is never filed as a different category; it's sent back for
  manual routing.

## Failure handling

| Situation | Where it's caught | Result |
|---|---|---|
| Specialist not running | Requester (agent card / submit) | `Specialist Agent is unavailable` |
| Specialist takes too long | Requester polling timeout | `TIMEOUT ... (last status: working)`, no ticket |
| Question unrelated to the KB | Retriever similarity gate (< 0.2) | Task `failed`: `insufficient_context` |
| Passages retrieved but don't answer it | LLM returns `answerable: false` | Task `failed`: `insufficient_context` |
| `GROQ_API_KEY` missing / invalid | Generator | Task `failed`: `missing_api_key` / `llm_auth_error` |
| Groq rate limit, network error, or 5xx | Generator (after one SDK retry) | Task `failed`: `llm_unavailable` |
| LLM returns malformed JSON | Generator (one retry) | Task `failed`: `llm_invalid_output` |
| Unexpected Specialist exception | A2A server | Task `failed`: `internal_error` |
| Unknown task ID | A2A server | HTTP 404 |
| Category not offered by the form (Email, Security) | Requester validation | `CANNOT FILE TICKET ... Route this ticket manually` |
| Form shows an error or the confirmation doesn't match | Playwright verification | `Ticket submission did not verify` |
| Browser can't launch (Chromium not installed) | Playwright launch | `Could not launch the browser ... Run playwright install chromium` |
| Any unforeseen error during a request | Interactive session safety net | `UNEXPECTED ERROR: ...`; the session continues with the next request |

`--simulate slow` / `/slow` makes the Specialist sleep 60 s; `--simulate fail` /
`/fail` makes it fail immediately. These make the failure paths easy to
reproduce.

**Known limitations:** tasks are kept in memory, so they're lost when the
Specialist restarts. A task keeps running on the Specialist after the Requester
times out (there's no cancel endpoint). The Requester's list of form categories
is a constant; Playwright re-checks it against the live page before
submitting.

## Testing components individually

| Command | What it does |
|---|---|
| `python -m rag.ingest --check` | Rebuild the index and run `test_cases.json` through retrieval |
| `python -m rag.ingest --compare` | Same, plus a side-by-side run without contextual headers |
| `python -m rag.pipeline "My monitor is flickering"` | Full RAG answer as JSON (no agents) |
| `python -m rag.pipeline "..." --retrieval-only` | Retrieved + reranked chunks only (no LLM / API key) |
| `python -m automation.ticket_form` | Submit one hard-coded ticket with Playwright (no agents) |
| `python -m automation.ticket_form --category Email` | Unsupported-category failure |
| `python -m automation.app_server` | Serve the mock app at http://localhost:8000/index.html |

## Project structure

```
a2a_protocol/          A2A-style protocol, independent of RAG
  models.py            Task, TaskRequest, SpecialistResult, statuses
  server.py            FastAPI task server: submit, status, agent card
  client.py            Submit + poll with timeout; typed errors
specialist/            Specialist Agent
  agent.py             handle_task(): runs RAG, maps errors to failed tasks
  __main__.py          python -m specialist
requester/             Requester Agent (rule-based coordinator)
  agent.py             run(): submit -> poll -> validate -> Playwright
  __main__.py          python -m requester (interactive or single request)
rag/                   RAG pipeline
  config.py            Models, paths, thresholds
  ingest.py            Load, chunk, add headers, embed, save
  vector_store.py      Embedding model + FAISS load/save
  retriever.py         Vector search, relevance gate, cross-encoder rerank
  generator.py         Groq call with strict JSON schema
  pipeline.py          answer_question(): retrieve -> generate
  errors.py            Error codes surfaced as failed tasks
automation/            Playwright workflow
  ticket_form.py       Fill, submit, verify the ticket form
  app_server.py        Serve the mock app over HTTP
knowledge_base/        Course-provided support documents
mock_support_app/      Course-provided ticket form (unchanged)
test_cases.json        Course-provided test requests
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` (e.g. `langchain_community`) | The venv isn't active. Activate it, or use `.venv/Scripts/python` (Windows) / `.venv/bin/python` (macOS/Linux) |
| PowerShell: "running scripts is disabled" when activating | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or skip activation and use `.venv/Scripts/python` |
| `No vector store found` | Run `python -m rag.ingest` |
| Task fails with `missing_api_key` | Create `.env` in the project folder (see Setup step 3) and restart the Specialist |
| Task fails with `llm_unavailable` after many requests | Groq's free tier is rate-limited per minute. Wait a minute and try again |
| `Specialist Agent is unavailable` | Start `python -m specialist` in another terminal first |
| Port 8001 already in use | Another Specialist is still running. Stop it, or use `python -m specialist --port 9001` with `python -m requester --specialist-url http://127.0.0.1:9001` |
| Browser doesn't open | Run `playwright install chromium` |
