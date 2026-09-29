"""FastAPI application exposing the agent to the dashboard.

Run with: uvicorn backend.main:app --reload
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.agent.graph import get_compiled_graph, run_agent
from backend.agent.tools import load_appointments, load_escalations
from backend.config import settings
from backend.memory.session import Session, session_store
from backend.models.schemas import (
    AnalyticsResponse,
    Appointment,
    ChatRequest,
    ChatResponse,
    Escalation,
    HistoryResponse,
    IngestResponse,
    MessageModel,
)
from backend.retrieval.ingest import ingest

settings.configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Compile the agent graph at startup so the first chat is not slow."""
    get_compiled_graph()
    logger.info("API ready for %s", settings.business_name)
    yield


app = FastAPI(
    title=f"{settings.business_name} Voice Agent",
    description="Chat agent with hybrid retrieval, booking, and escalation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Report that the API is up and whether a Groq key is configured."""
    key_set = bool(settings.groq_api_key) and settings.groq_api_key != "your_groq_api_key_here"
    return {
        "status": "ok",
        "business_name": settings.business_name,
        "model": settings.groq_model,
        "groq_api_key_configured": key_set,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Handle one caller message and return the agent's reply."""
    session = session_store.add_user_message(request.session_id, request.message)
    window = session_store.context_window(request.session_id)

    state = await run_agent(
        session_id=request.session_id,
        user_input=request.message,
        messages=window,
        booking_info=session.booking_info,
    )

    response_text = state.get("response", "")
    intent = state.get("intent", "unknown")
    confidence = float(state.get("confidence", 0.0))
    action_taken = state.get("action_taken", "none")
    booking_info = state.get("booking_info", {}) or {}

    if action_taken == "book_appointment":
        # The visit is booked, so the next request starts from scratch.
        booking_info = {}
        session.booking_info.clear()

    session_store.add_agent_turn(
        session_id=request.session_id,
        response=response_text,
        intent=intent,
        action_taken=action_taken,
        confidence=confidence,
        booking_info=booking_info,
    )

    return ChatResponse(
        response=response_text,
        intent=intent,
        confidence=max(0.0, min(1.0, confidence)),
        action_taken=action_taken,
        needs_escalation=bool(state.get("needs_escalation", False)),
        session_id=request.session_id,
    )


@app.get("/sessions/{session_id}/history", response_model=HistoryResponse)
async def get_history(session_id: str) -> HistoryResponse:
    """Return the full stored conversation for one session."""
    session: Session | None = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session '{session_id}'.")
    return HistoryResponse(
        session_id=session.session_id,
        messages=[MessageModel(role=m["role"], content=m["content"]) for m in session.messages],
        intent_history=session.intent_history,
        actions_taken=session.actions_taken,
        summary=session.summary,
        started_at=session.started_at,
        last_active_at=session.last_active_at,
        turns=session.turns,
    )


@app.get("/appointments", response_model=list[Appointment])
async def get_appointments() -> list[Appointment]:
    """List every appointment booked through the agent, newest first."""
    records = await asyncio.to_thread(load_appointments)
    return [Appointment(**record) for record in reversed(records)]


@app.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics() -> AnalyticsResponse:
    """Aggregate session, booking, and escalation stats for the dashboard."""
    sessions = session_store.all_sessions()
    appointments = await asyncio.to_thread(load_appointments)
    escalations = await asyncio.to_thread(load_escalations)
    today = date.today().isoformat()

    intent_breakdown: dict[str, int] = {}
    confidences: list[float] = []
    total_messages = 0
    for session in sessions:
        total_messages += len(session.messages)
        confidences.extend(session.confidences)
        for intent in session.intent_history:
            intent_breakdown[intent] = intent_breakdown.get(intent, 0) + 1

    conversations_today = sum(
        1 for session in sessions if session.started_at.date().isoformat() == today
    )
    escalation_turns = sum(
        1
        for session in sessions
        for action in session.actions_taken
        if action == "escalate_to_human"
    )
    total_turns = sum(len(session.intent_history) for session in sessions)

    return AnalyticsResponse(
        total_conversations=len(sessions),
        conversations_today=conversations_today,
        total_messages=total_messages,
        intent_breakdown=intent_breakdown,
        average_confidence=(sum(confidences) / len(confidences)) if confidences else 0.0,
        total_appointments=len(appointments),
        appointments_today=sum(1 for a in appointments if a.get("date") == today),
        total_escalations=len(escalations),
        escalation_rate=(escalation_turns / total_turns) if total_turns else 0.0,
        recent_appointments=[Appointment(**record) for record in list(reversed(appointments))[:10]],
        recent_escalations=[Escalation(**record) for record in list(reversed(escalations))[:10]],
    )


@app.post("/ingest", response_model=IngestResponse)
async def trigger_ingest() -> IngestResponse:
    """Re-index the knowledge base from data/knowledge_base/."""
    try:
        summary = await asyncio.to_thread(ingest)
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
    return IngestResponse(
        documents=int(summary["documents"]),
        chunks=int(summary["chunks"]),
        embedding_model=str(summary["embedding_model"]),
        chroma_persist_dir=str(summary["chroma_persist_dir"]),
        bm25_index_path=str(summary["bm25_index_path"]),
    )
