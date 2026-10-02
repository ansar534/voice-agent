"""Pydantic models validating every API request and response."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """One caller message. An empty session_id starts a new conversation."""

    session_id: str = Field(default="", max_length=128)
    message: str = Field(..., min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    """The agent's reply plus what it decided along the way."""

    response: str
    intent: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    action_taken: str
    session_id: str
    needs_escalation: bool = False


class HistoryResponse(BaseModel):
    """Everything stored about one conversation."""

    session_id: str
    messages: list[dict]
    intent_history: list[str]
    actions_taken: list[str]
    turn_count: int
    start_time: datetime
    last_active: datetime


class Appointment(BaseModel):
    """A booking saved to appointments.json."""

    booking_id: str
    name: str
    date: str
    time: str
    reason: str
    created_at: str


class Escalation(BaseModel):
    """A handoff saved to escalations.json."""

    escalation_id: str
    reason: str
    summary: str
    timestamp: str


class AnalyticsResponse(BaseModel):
    """Aggregate numbers shown on the analytics tab."""

    total_sessions: int
    intent_counts: dict[str, int]
    action_counts: dict[str, int]
    top_intent: str
    avg_turns: float
    escalation_rate: float
    avg_confidence: float
    total_messages: int
    total_appointments: int
    total_escalations: int


class IngestResponse(BaseModel):
    """Result of rebuilding the knowledge base indexes."""

    status: str
    documents: int
    chunks_stored: int
    embedding_model: str


class HealthResponse(BaseModel):
    """Service status and configuration summary."""

    status: str
    business: str
    model: str
    groq_key_configured: bool
