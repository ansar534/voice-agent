"""State passed between nodes of the agent graph.

LangGraph merges the dict each node returns into this state, so a node
only returns the keys it changed.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

Intent = Literal["faq", "book_appointment", "escalate", "unknown"]

VALID_INTENTS: tuple[Intent, ...] = ("faq", "book_appointment", "escalate", "unknown")


class Message(TypedDict):
    """One turn of the conversation as stored in session memory."""

    role: Literal["user", "assistant"]
    content: str


class BookingInfo(TypedDict, total=False):
    """Appointment fields gathered across turns. Any field may be missing."""

    name: str
    date: str
    time: str
    reason: str


class AgentState(TypedDict, total=False):
    """Everything the graph reads or writes while handling one user turn."""

    # Conversation so far, including the current user message.
    messages: list[Message]
    # The message being handled on this pass through the graph.
    user_input: str
    # Intent chosen by classify_intent.
    intent: Intent
    # Chunks returned by hybrid retrieval, newest query only.
    retrieved_context: list[dict[str, Any]]
    # Cosine similarity of the best retrieved chunk, 0 to 1.
    confidence: float
    # Identifies the conversation this turn belongs to.
    session_id: str
    # True once confidence is too low or the caller asked for a human.
    needs_escalation: bool
    # Name of the tool or node that acted, for the dashboard.
    action_taken: str
    # Reply shown to the caller.
    response: str
    # Appointment fields collected so far.
    booking_info: BookingInfo
    # Guards the unknown -> clarify -> classify_intent loop.
    clarification_attempts: int
    # Set when a node fails, so the API can report it without crashing.
    error: str


def new_state(session_id: str, user_input: str, messages: list[Message]) -> AgentState:
    """Build the starting state for one user turn with safe defaults."""
    return AgentState(
        messages=messages,
        user_input=user_input,
        intent="unknown",
        retrieved_context=[],
        confidence=0.0,
        session_id=session_id,
        needs_escalation=False,
        action_taken="none",
        response="",
        booking_info={},
        clarification_attempts=0,
        error="",
    )
