# AI Voice Agent — Complete (Phases 1–4)

Alex is an AI receptionist for **City Medical Clinic**. It answers questions
from a local knowledge base, books appointments, speaks over voice and phone,
and runs post-call automation with analytics.

| Phase | Capability |
|---|---|
| 1 | Chat + hybrid RAG + LangGraph agent |
| 2 | Voice streaming (Deepgram STT + ElevenLabs TTS) |
| 3 | Phone calls (Twilio webhooks) |
| 4 | Post-call automation + advanced analytics |

## Stack

| Piece | Choice |
|---|---|
| API | FastAPI + uvicorn |
| Orchestration | LangGraph `StateGraph` |
| LLM | Groq (`openai/gpt-oss-20b` by default) |
| Retrieval | ChromaDB (semantic) + rank_bm25 (keyword) |
| Embeddings | `all-MiniLM-L6-v2` (local) |
| STT | Deepgram nova-2 |
| TTS | ElevenLabs |
| Phone / SMS | Twilio |
| Dashboard | Streamlit |

## Setup

Requires Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in keys:

```
GROQ_API_KEY=
DEEPGRAM_API_KEY=
ELEVENLABS_API_KEY=
ELEVENLABS_VOICE_ID=Rachel
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_PHONE_NUMBER=
BUSINESS_NAME=City Medical Clinic
CHROMA_PERSIST_DIR=./chroma_db
KNOWLEDGE_BASE_DIR=./data/knowledge_base
BM25_INDEX_PATH=./bm25_index.pkl
LOG_LEVEL=INFO
CONFIDENCE_THRESHOLD=0.4
MAX_HISTORY_TURNS=6
```

Free key consoles: [Groq](https://console.groq.com), [Deepgram](https://console.deepgram.com),
[ElevenLabs](https://elevenlabs.io), [Twilio](https://www.twilio.com) (trial credit).

## How to run

**1. Index the knowledge base** (once, and after KB changes):

```bash
python -m backend.retrieval.ingest
```

**2. Start the API** (terminal 1):

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**3. Start the dashboard** (terminal 2):

```bash
streamlit run frontend/app.py
```

Open http://localhost:8501 — Chat, Phone Settings, Voice Chat, Advanced Analytics.

## Testing order

1. Start with only `GROQ_API_KEY` → test **Chat**
2. Add Deepgram → test **Voice Chat** transcription
3. Add ElevenLabs → hear spoken replies
4. Add Twilio → set webhook to `/phone/incoming`, make a test call
5. Review **Advanced Analytics**

## Phone setup

1. Buy/get a free Twilio number
2. Voice webhook: `https://your-domain.com/phone/incoming` (POST)
3. Optional recording callback: `https://your-domain.com/phone/record`
4. Forward the clinic number to the Twilio number

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/chat` | Text chat turn |
| GET | `/sessions/{id}/history` | Conversation history |
| GET | `/appointments` | Bookings |
| GET | `/escalations` | Human handoffs |
| GET | `/analytics` | Full dashboard metrics |
| POST | `/ingest` | Rebuild KB indexes |
| GET | `/health` | Status + key configuration |
| POST | `/audio/transcribe` | Deepgram file STT |
| POST | `/audio/speak` | ElevenLabs TTS (mp3) |
| POST | `/audio/chat` | Voice turn (STT → agent → TTS) |
| POST | `/phone/incoming` | Twilio inbound call webhook |
| POST | `/phone/gather` | Twilio Gather speech callback |
| POST | `/phone/record` | Voicemail / recording webhook |
| GET | `/calls` | Call log |
| GET/PUT | `/phone/settings` | Phone agent toggles |
| POST | `/phone/test-call` | Outbound Twilio test |
| GET | `/analytics/transcripts.csv` | Download call transcripts |

## Agent flow

```
classify_intent
  faq              -> retrieve_context -> generate_response -> END
                                       -> handle_escalation -> END  (confidence < 0.4)
  book_appointment -> collect_booking_info -> process_booking -> END
  escalate         -> handle_escalation -> END
  unknown          -> handle_unknown -> classify_intent (one retry) -> END
```

## Tests

```bash
python -m pytest tests -q
```

Run ingestion before pytest so retrieval tests can hit the indexes.
