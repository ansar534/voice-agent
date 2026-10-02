"""Shared state that flows through the agent graph."""

from __future__ import annotations

import operator
from typing import Annotated, List, TypedDict


class AgentState(TypedDict, total=False):
    """One turn's working state.

    ``messages`` uses operator.add as its reducer, so a node returns only
    the NEW messages it produced and LangGraph appends them. Returning the
    whole list from a node would duplicate the history.
    """

    messages: Annotated[List[dict], operator.add]
    session_id: str
    # One of: "faq" | "book_appointment" | "escalate" | "unknown"
    intent: str
    retrieved_context: str
    confidence: float
    needs_escalation: bool
    action_taken: str
    # Booking fields gathered across turns: name, date, time, reason.
    booking_info: dict
    turn_count: int
    # Caps the unknown -> clarify -> classify loop so it cannot spin forever.
    clarify_attempts: int


VALID_INTENTS = ("faq", "book_appointment", "escalate", "unknown")
BOOKING_FIELDS = ("name", "date", "time", "reason")
