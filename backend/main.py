"""FastAPI service exposing chat, voice, phone, and analytics.

Start with:
    python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import base64
import csv
import io
import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agent.graph import run_agent
from .agent.tools import get_appointments, get_escalations
from .analytics import build_analytics
from .audio.speaker import get_speaker
from .audio.transcriber import get_transcriber
from .config import (
    AUDIO_CACHE_DIR,
    BUSINESS_NAME,
    GROQ_MODEL,
    PUBLIC_BASE_URL,
    TWILIO_PHONE_NUMBER,
    configure_logging,
    deepgram_key_is_set,
    elevenlabs_key_is_set,
    groq_key_is_set,
    twilio_configured,
)
from .memory.session import session_memory
from .models.schemas import (
    AnalyticsResponse,
    Appointment,
    ChatRequest,
    ChatResponse,
    Escalation,
    HealthResponse,
    HistoryResponse,
    IngestResponse,
    PhoneSettings,
    SpeakRequest,
    TestCallRequest,
    VoiceChatResponse,
)
from .phone.twilio_handler import (
    get_calls,
    get_phone_agent,
    load_phone_settings,
    save_phone_settings,
)
from .retrieval.ingest import run_ingestion
from .retrieval.vectorstore import reset_retriever

configure_logging()
logger = logging.getLogger(__name__)

AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title=f"{BUSINESS_NAME} Voice Agent",
    description=(
        "Chat agent with hybrid retrieval, voice streaming, Twilio phone calls, "
        "and post-call automation."
    ),
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/audio/files", StaticFiles(directory=str(AUDIO_CACHE_DIR)), name="audio_files")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log the method, path, status, and duration of every request."""
    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "%s %s -> %d (%.0f ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Phase 1 — Chat + RAG + Agent
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Report service status and which integrations are configured."""
    return HealthResponse(
        status="ok",
        business=BUSINESS_NAME,
        model=GROQ_MODEL,
        groq_key_configured=groq_key_is_set(),
        deepgram_key_configured=deepgram_key_is_set(),
        elevenlabs_key_configured=elevenlabs_key_is_set(),
        twilio_configured=twilio_configured(),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Handle one caller message and return the agent's reply."""
    session_id = request.session_id.strip() or str(uuid.uuid4())
    try:
        result = await asyncio.to_thread(run_agent, session_id, request.message)
    except Exception as exc:
        logger.exception("Chat failed for session %s", session_id)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    return ChatResponse(
        response=result["response"],
        intent=result["intent"],
        confidence=max(0.0, min(1.0, float(result["confidence"]))),
        action_taken=result["action_taken"],
        session_id=result["session_id"],
        needs_escalation=bool(result.get("needs_escalation", False)),
    )


@app.get("/sessions/{session_id}/history", response_model=HistoryResponse)
async def get_session_history(session_id: str) -> HistoryResponse:
    """Return the full message history and metadata for one session."""
    if session_id not in session_memory.sessions:
        raise HTTPException(status_code=404, detail=f"Unknown session '{session_id}'.")

    session = session_memory.get_session(session_id)
    return HistoryResponse(
        session_id=session_id,
        messages=session_memory.get_full_history(session_id),
        intent_history=session["intent_history"],
        actions_taken=session["actions_taken"],
        turn_count=session["turn_count"],
        start_time=session["start_time"],
        last_active=session["last_active"],
    )


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    """Forget one conversation."""
    if not session_memory.clear_session(session_id):
        raise HTTPException(status_code=404, detail=f"Unknown session '{session_id}'.")
    return {"status": "cleared", "session_id": session_id}


@app.get("/appointments", response_model=list[Appointment])
async def list_appointments() -> list[Appointment]:
    """Return every booked appointment, newest first."""
    records = await asyncio.to_thread(get_appointments)
    return [Appointment(**record) for record in reversed(records)]


@app.get("/escalations", response_model=list[Escalation])
async def list_escalations() -> list[Escalation]:
    """Return every logged handoff to a human, newest first."""
    records = await asyncio.to_thread(get_escalations)
    return [Escalation(**record) for record in reversed(records)]


@app.get("/analytics")
async def analytics() -> dict:
    """Return chat, call, retrieval, performance, and cost analytics."""
    return await asyncio.to_thread(build_analytics)


@app.post("/ingest", response_model=IngestResponse)
async def ingest() -> IngestResponse:
    """Rebuild the knowledge base indexes from data/knowledge_base/."""
    try:
        summary = await asyncio.to_thread(run_ingestion)
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    reset_retriever()
    return IngestResponse(
        status="success",
        documents=int(summary["documents"]),
        chunks_stored=int(summary["chunks_stored"]),
        embedding_model=str(summary["embedding_model"]),
    )


# ---------------------------------------------------------------------------
# Phase 2 — Voice streaming (Deepgram + ElevenLabs)
# ---------------------------------------------------------------------------
@app.post("/audio/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)) -> dict:
    """Transcribe an uploaded audio file with Deepgram."""
    if not deepgram_key_is_set():
        raise HTTPException(status_code=503, detail="DEEPGRAM_API_KEY is not configured.")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload.")

    try:
        transcript, confidence = await asyncio.to_thread(
            get_transcriber().transcribe_file_with_confidence, audio_bytes
        )
    except Exception as exc:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=502, detail=f"Deepgram error: {exc}") from exc

    return {"transcript": transcript, "confidence": confidence}


@app.post("/audio/speak")
async def speak(request: SpeakRequest) -> Response:
    """Convert text to speech and return raw mp3 audio."""
    if not elevenlabs_key_is_set():
        raise HTTPException(status_code=503, detail="ELEVENLABS_API_KEY is not configured.")

    try:
        audio_bytes = await asyncio.to_thread(get_speaker().generate_speech_sync, request.text)
    except Exception as exc:
        logger.exception("Speech generation failed")
        raise HTTPException(status_code=502, detail=f"ElevenLabs error: {exc}") from exc

    return Response(content=audio_bytes, media_type="audio/mpeg")


@app.post("/audio/chat", response_model=VoiceChatResponse)
async def audio_chat(
    audio: UploadFile = File(...),
    session_id: str = Form(default=""),
) -> VoiceChatResponse:
    """Full voice turn: transcribe → agent → speak."""
    if not deepgram_key_is_set():
        raise HTTPException(status_code=503, detail="DEEPGRAM_API_KEY is not configured.")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload.")

    try:
        transcript = await asyncio.to_thread(get_transcriber().transcribe_file, audio_bytes)
    except Exception as exc:
        logger.exception("Voice chat transcription failed")
        raise HTTPException(status_code=502, detail=f"Deepgram error: {exc}") from exc

    if not transcript.strip():
        raise HTTPException(
            status_code=422,
            detail="Sorry, I didn't catch that. Try again.",
        )

    sid = session_id.strip() or str(uuid.uuid4())
    try:
        result = await asyncio.to_thread(run_agent, sid, transcript)
    except Exception as exc:
        logger.exception("Voice chat agent failed")
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    audio_b64 = ""
    if elevenlabs_key_is_set():
        try:
            spoken = await asyncio.to_thread(
                get_speaker().generate_speech_sync, result["response"]
            )
            audio_b64 = base64.b64encode(spoken).decode("ascii")
        except Exception as exc:
            logger.warning("ElevenLabs failed during voice chat: %s", exc)

    return VoiceChatResponse(
        transcript=transcript,
        response=result["response"],
        audio=audio_b64,
        intent=result["intent"],
        confidence=max(0.0, min(1.0, float(result["confidence"]))),
        action_taken=result["action_taken"],
        session_id=result["session_id"],
        needs_escalation=bool(result.get("needs_escalation", False)),
    )


# ---------------------------------------------------------------------------
# Phase 3 — Phone calls (Twilio)
# ---------------------------------------------------------------------------
@app.post("/phone/incoming")
async def phone_incoming(request: Request) -> Response:
    """Twilio webhook for inbound calls."""
    form = await request.form()
    from_number = str(form.get("From", ""))
    to_number = str(form.get("To", TWILIO_PHONE_NUMBER or ""))
    twiml = await get_phone_agent().handle_incoming_call(from_number, to_number)
    return Response(content=twiml, media_type="application/xml")


@app.post("/phone/gather")
async def phone_gather(request: Request) -> Response:
    """Twilio Gather callback: speech → agent → spoken reply."""
    form = await request.form()
    session_id = str(request.query_params.get("session_id") or form.get("session_id") or "")
    if not session_id:
        session_id = f"phone-{uuid.uuid4().hex[:12]}"

    speech_result = str(form.get("SpeechResult", "") or "")
    call_sid = str(form.get("CallSid", "") or "")
    from_number = str(form.get("From", "") or "")
    confidence_raw = form.get("Confidence")
    confidence = float(confidence_raw) if confidence_raw not in (None, "") else None

    twiml = await get_phone_agent().handle_gather(
        session_id=session_id,
        speech_result=speech_result,
        call_sid=call_sid,
        from_number=from_number,
        confidence=confidence,
    )
    return Response(content=twiml, media_type="application/xml")


@app.post("/phone/record")
async def phone_record(request: Request) -> JSONResponse:
    """Twilio webhook for voicemail / call recordings."""
    form = await request.form()
    recording_url = str(form.get("RecordingUrl", "") or "")
    call_sid = str(form.get("CallSid", "") or "")
    from_number = str(form.get("From", "") or "")
    duration = int(form.get("RecordingDuration") or form.get("CallDuration") or 0)

    if not recording_url:
        return JSONResponse({"status": "ignored", "reason": "no recording url"})

    result = await get_phone_agent().handle_recording(
        call_sid=call_sid,
        recording_url=recording_url,
        from_number=from_number,
        duration=duration,
    )
    return JSONResponse(result)


@app.get("/calls")
async def list_calls() -> list[dict]:
    """Return every phone call logged by the agent."""
    return await asyncio.to_thread(get_calls)


@app.get("/phone/settings", response_model=PhoneSettings)
async def get_phone_settings() -> PhoneSettings:
    """Return the current phone agent settings."""
    return PhoneSettings(**load_phone_settings())


@app.put("/phone/settings", response_model=PhoneSettings)
async def update_phone_settings(settings: PhoneSettings) -> PhoneSettings:
    """Persist phone agent settings from the dashboard."""
    saved = save_phone_settings(settings.model_dump())
    return PhoneSettings(**saved)


@app.post("/phone/test-call")
async def phone_test_call(payload: TestCallRequest) -> dict:
    """Place an outbound Twilio test call."""
    if not twilio_configured():
        raise HTTPException(status_code=503, detail="Twilio is not configured.")
    try:
        return await asyncio.to_thread(get_phone_agent().make_test_call, payload.to_number)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/phone/info")
async def phone_info() -> dict:
    """Return the configured Twilio number and public webhook URL."""
    return {
        "phone_number": TWILIO_PHONE_NUMBER,
        "configured": twilio_configured(),
        "incoming_webhook": f"{PUBLIC_BASE_URL}/phone/incoming",
        "record_webhook": f"{PUBLIC_BASE_URL}/phone/record",
        "settings": load_phone_settings(),
    }


# ---------------------------------------------------------------------------
# Phase 4 — Exports
# ---------------------------------------------------------------------------
@app.get("/analytics/transcripts.csv")
async def download_transcripts_csv() -> StreamingResponse:
    """Download call transcripts as CSV for compliance archives."""

    def _build() -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            ["call_sid", "from_number", "timestamp", "duration", "status", "transcript"]
        )
        for call in get_calls():
            writer.writerow(
                [
                    call.get("call_sid", ""),
                    call.get("from_number", ""),
                    call.get("timestamp", ""),
                    call.get("duration", ""),
                    call.get("status", ""),
                    call.get("transcript", ""),
                ]
            )
        return buffer.getvalue()

    content = await asyncio.to_thread(_build)
    return StreamingResponse(
        iter([content]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=call_transcripts.csv"},
    )
