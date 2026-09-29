"""Request and response models for every API endpoint."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """A single message from a caller in an ongoing session."""

    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    """What the agent decided and what the caller should hear."""

    response: str
    intent: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    action_taken: str
    needs_escalation: bool = False
    session_id: str


class MessageModel(BaseModel):
    """One stored conversation turn."""

    role: str
    content: str


class HistoryResponse(BaseModel):
    """Full conversation history for one session."""

    session_id: str
    messages: list[MessageModel]
    intent_history: list[str]
    actions_taken: list[str]
    summary: str = ""
    started_at: datetime
    last_active_at: datetime
    turns: int


class Appointment(BaseModel):
    """A booking saved to the mock CRM."""

    booking_id: str
    name: str
    date: str
    time: str
    reason: str
    created_at: str


class Escalation(BaseModel):
    """A logged handoff to a human."""

    escalation_id: str
    reason: str
    summary: str
    created_at: str


class AnalyticsResponse(BaseModel):
    """Aggregate numbers shown on the dashboard."""

    total_conversations: int
    conversations_today: int
    total_messages: int
    intent_breakdown: dict[str, int]
    average_confidence: float
    total_appointments: int
    appointments_today: int
    total_escalations: int
    escalation_rate: float
    recent_appointments: list[Appointment]
    recent_escalations: list[Escalation]


class IngestResponse(BaseModel):
    """Result of re-indexing the knowledge base."""

    documents: int
    chunks: int
    embedding_model: str
    chroma_persist_dir: str
    bm25_index_path: str
