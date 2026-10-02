# AI Voice Agent — Phase 1 (Chat)

Alex is an AI receptionist for a medical clinic. It answers questions from
a local knowledge base, books appointments, and hands off to a human when
it isn't confident enough to answer. Phase 1 is chat only; voice and phone
come later.

## Stack

| Piece | Choice |
|---|---|
| API | FastAPI + uvicorn |
| Orchestration | LangGraph `StateGraph` |
| LLM | Groq, `openai/gpt-oss-20b` (set `GROQ_MODEL` to change) |
| Retrieval | ChromaDB (semantic) + rank_bm25 (keyword) |
| Embeddings | `all-MiniLM-L6-v2`, runs locally, no API key |
| Dashboard | Streamlit |

## Setup

Requires Python 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Groq key from
[console.groq.com](https://console.groq.com/keys):

```
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-20b
BUSINESS_NAME=City Medical Clinic
CHROMA_PERSIST_DIR=./chroma_db
KNOWLEDGE_BASE_DIR=./data/knowledge_base
BM25_INDEX_PATH=./bm25_index.pkl
LOG_LEVEL=INFO
CONFIDENCE_THRESHOLD=0.4
MAX_HISTORY_TURNS=6
```

Groq retired the Llama 3.1 8B endpoints, so the default is
`openai/gpt-oss-20b`. To see what your key can use:
`GET https://api.groq.com/openai/v1/models`.

## Running

Run these in order, in two terminals.

**1. Index the knowledge base.** Required once, and again whenever files in
`data/knowledge_base/` change:

```powershell
.\.venv\Scripts\python.exe -m backend.retrieval.ingest
```

**2. Start the API** (terminal 1):

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

**3. Start the dashboard** (terminal 2):

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

The dashboard opens at http://localhost:8501 and the interactive API docs
are at http://localhost:8000/docs.

## How a turn flows

```
classify_intent
  faq              -> retrieve_context -> generate_response -> END
                                       -> handle_escalation -> END  (confidence < 0.4)
  book_appointment -> collect_booking_info -> process_booking -> END
                                           -> END                   (still collecting)
  escalate         -> handle_escalation -> END
  unknown          -> handle_unknown -> classify_intent (one retry) -> END
```

Retrieval runs semantic and keyword search together and blends them
`0.6 * semantic + 0.4 * BM25`. Confidence is the cosine similarity between
the question and the best chunk. Below `CONFIDENCE_THRESHOLD` the agent
hands off to staff instead of guessing, and answers are generated only
from retrieved passages.

The `unknown` branch loops back to `classify_intent` once. The retry is
capped because re-classifying the same text with no new input would
otherwise never terminate.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/chat` | Send a message; empty `session_id` starts a new conversation |
| GET | `/sessions/{session_id}/history` | Full conversation history |
| DELETE | `/sessions/{session_id}` | Forget a conversation |
| GET | `/appointments` | Every booking |
| GET | `/escalations` | Every handoff to a human |
| GET | `/analytics` | Dashboard stats |
| POST | `/ingest` | Rebuild the knowledge base indexes |
| GET | `/health` | Status and whether a Groq key is configured |

## Data

Knowledge base documents are `.txt` files in `data/knowledge_base/`.
Bookings go to `appointments.json` and handoffs to `escalations.json`,
both standing in for a real CRM. Session memory is in-process, so
restarting the API clears conversations but not the JSON files.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

The tests script the LLM so they run offline and deterministically, but
they use the real ChromaDB and BM25 indexes. Run ingestion first.
