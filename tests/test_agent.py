"""Agent behaviour tests.

The Groq call is replaced with a scripted stand-in so these run offline
and give the same answer every time. Retrieval is real: the tests read
the ChromaDB and BM25 indexes, so run ingestion before pytest.
"""

from __future__ import annotations

import json

import pytest

from backend.agent import nodes, tools
from backend.agent.graph import run_agent
from backend.config import CONFIDENCE_THRESHOLD
from backend.memory.session import session_memory


def fake_call_llm(messages: list[dict], temperature: float, purpose: str) -> str:
    """Return a canned reply based on which node is calling.

    Mirrors how a small model would behave on these inputs, without the
    network round trip or the run-to-run variation.
    """
    last_content = str(messages[-1].get("content", "")).lower()

    if purpose == "classify_intent":
        if any(word in last_content for word in ("idiot", "useless", "human", "manager")):
            return "escalate"
        if any(word in last_content for word in ("appointment", "book", "schedule")):
            return "book_appointment"
        if "?" in last_content:
            return "faq"
        return "unknown"

    if purpose == "collect_booking_info":
        return json.dumps(
            {
                "name": "Maria Chen" if "maria chen" in last_content else "",
                "date": "2026-10-06" if "2026-10-06" in last_content else "",
                "time": "2:00 PM" if "2:00 pm" in last_content else "",
                "reason": "blood pressure check" if "blood pressure" in last_content else "",
            }
        )

    if purpose == "generate_response":
        return "Saturday hours are 9:00 AM to 1:00 PM for urgent same-week visits."

    if purpose == "escalation_summary":
        return "The caller needs help the knowledge base cannot provide. Staff follow-up required."

    return "How can I help?"


@pytest.fixture(autouse=True)
def isolated_agent(monkeypatch, tmp_path):
    """Script the LLM, redirect JSON writes to tmp, and clear memory."""
    monkeypatch.setattr(nodes, "call_llm", fake_call_llm)
    monkeypatch.setattr(tools, "APPOINTMENTS_PATH", tmp_path / "appointments.json")
    monkeypatch.setattr(tools, "ESCALATIONS_PATH", tmp_path / "escalations.json")
    session_memory.clear_all()
    yield
    session_memory.clear_all()


def test_faq_with_answer_in_knowledge_base_is_confident():
    """A question the documents cover is answered from retrieval."""
    result = run_agent("test-faq", "What are your Saturday hours?")

    assert result["intent"] == "faq"
    assert result["confidence"] >= CONFIDENCE_THRESHOLD
    assert result["action_taken"] == "faq_response"
    assert not result["needs_escalation"]
    assert "9:00" in result["response"]


def test_faq_without_answer_escalates_instead_of_guessing():
    """An off-topic question scores too low and hands off to a human."""
    result = run_agent("test-unknown-faq", "Who won the 1994 world series?")

    assert result["confidence"] < CONFIDENCE_THRESHOLD
    assert result["needs_escalation"]
    assert result["action_taken"] == "escalated"
    assert tools.get_escalations(), "the handoff should be logged"


def test_booking_happy_path_returns_booking_id():
    """A message with every detail books the visit and confirms it."""
    result = run_agent(
        "test-booking",
        "Book an appointment for Maria Chen on 2026-10-06 at 2:00 PM "
        "for a blood pressure check.",
    )

    assert result["intent"] == "book_appointment"
    assert result["action_taken"] == "appointment_booked"
    assert "CMC-" in result["response"]

    saved = tools.get_appointments()
    assert len(saved) == 1
    assert saved[0]["name"] == "Maria Chen"
    assert saved[0]["date"] == "2026-10-06"


def test_booking_without_date_asks_for_it():
    """Missing details produce a question, never a guessed booking."""
    result = run_agent(
        "test-partial-booking",
        "I'd like to book an appointment for Maria Chen for a blood pressure check.",
    )

    assert result["action_taken"] == "collecting_booking_info"
    assert "date" in result["response"].lower()
    assert tools.get_appointments() == []


def test_offensive_input_is_deflected_gracefully():
    """Abuse is handed to a human rather than answered in kind."""
    result = run_agent("test-abuse", "You people are useless idiots")

    assert result["needs_escalation"]
    assert result["action_taken"] == "escalated"
    assert result["response"].strip()
    assert "idiot" not in result["response"].lower()
