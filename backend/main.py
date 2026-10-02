"""FastAPI service exposing the agent to the Streamlit dashboard.

Start with:
    python -m uvicorn backend.main:app --reload
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .agent.graph import run_agent
from .agent.tools import get_appointments, get_escalations
from .config import BUSINESS_NAME, GROQ_MODEL, configure_logging, groq_key_is_set
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
)
from .retrieval.ingest import run_ingestion
from .retrieval.vectorstore import reset_retriever

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title=f"{BUSINESS_NAME} Voice Agent",
    description="Chat agent with hybrid retrieval, appointment booking, and escalation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Report service status and whether a Groq key is configured."""
    return HealthResponse(
        status="ok",
        business=BUSINESS_NAME,
        model=GROQ_MODEL,
        groq_key_configured=groq_key_is_set(),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Handle one caller message and return the agent's reply.

    An empty session_id starts a new conversation with a fresh UUID.
    The agent runs in a worker thread because the graph is synchronous.
    """
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


@app.get("/analytics", response_model=AnalyticsResponse)
async def analytics() -> AnalyticsResponse:
    """Return aggregate session stats plus booking and escalation counts."""
    stats = session_memory.get_analytics()
    appointments = await asyncio.to_thread(get_appointments)
    escalations = await asyncio.to_thread(get_escalations)
    return AnalyticsResponse(
        **stats,
        total_appointments=len(appointments),
        total_escalations=len(escalations),
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest() -> IngestResponse:
    """Rebuild the knowledge base indexes from data/knowledge_base/."""
    try:
        summary = await asyncio.to_thread(run_ingestion)
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    # The retriever holds the old collection and index in memory.
    reset_retriever()
    return IngestResponse(
        status="success",
        documents=int(summary["documents"]),
        chunks_stored=int(summary["chunks_stored"]),
        embedding_model=str(summary["embedding_model"]),
    )
