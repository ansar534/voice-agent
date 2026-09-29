# AI Voice Agent — Phase 1 (Chat)

Alex is a receptionist agent for a clinic. It answers questions from a local
knowledge base, books appointments, and hands off to a human when it is not
confident. Phase 1 is chat only; voice and phone come later.

## Stack

| Piece | Choice |
|---|---|
| API | FastAPI + uvicorn |
| Orchestration | LangGraph state machine |
| LLM | Groq, `openai/gpt-oss-20b` (set `GROQ_MODEL` to change) |
| Retrieval | ChromaDB (semantic) + rank_bm25 (keyword) |
| Embeddings | `all-MiniLM-L6-v2`, local, no API key |
| Dashboard | Streamlit |

## Setup

Requires Python 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your key:

```
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-20b
BUSINESS_NAME=City Medical Clinic
CHROMA_PERSIST_DIR=./chroma_db
LOG_LEVEL=INFO
```

Groq has retired the Llama 3.1 8B endpoints, so the default is
`openai/gpt-oss-20b`. Check what your key can use with
`GET https://api.groq.com/openai/v1/models`.

Index the knowledge base. Re-run this whenever files in
`data/knowledge_base/` change:

```powershell
.\.venv\Scripts\python.exe -m backend.retrieval.ingest
```

## Running

Start the API, then the dashboard in a second terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

The dashboard is at http://localhost:8501 and the API docs at
http://localhost:8000/docs.

## How a turn flows

```
START -> classify_intent
  faq              -> retrieve_context -> generate_response -> END
                                       -> escalate -> END      (confidence < 0.4)
  book_appointment -> collect_booking_info -> book_appointment -> END
                                           -> END              (missing details)
  escalate         -> escalate -> END
  unknown          -> clarify -> classify_intent (once) -> END
```

Retrieval runs semantic and keyword search together, merges them by
reciprocal rank fusion, and reports the top chunk's cosine similarity as
confidence. Below 0.4 the agent hands off instead of guessing, and answers
are generated only from retrieved passages.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/chat` | Send a message, get a reply plus intent and confidence |
| GET | `/sessions/{session_id}/history` | Full conversation history |
| GET | `/appointments` | Every booking |
| GET | `/analytics` | Dashboard stats |
| POST | `/ingest` | Re-index the knowledge base |
| GET | `/health` | Status and whether a Groq key is configured |

## Data

Knowledge base documents live in `data/knowledge_base/` as `.txt`, `.md`,
or `.pdf`. Bookings are written to `appointments.json` and handoffs to
`escalations.json`, both standing in for a real CRM. Session memory is
in-process: the last 6 messages go to the model, and after 20 turns the
older ones are summarized.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

The tests script the LLM so they run offline, but they use the real Chroma
and BM25 indexes, so ingest before running them.
