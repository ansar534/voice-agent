"""Agent behaviour tests.

The Groq client is replaced with a scripted stand-in so these run offline
and give the same result every time. Retrieval is real: the tests read the
Chroma and BM25 indexes built by ingestion, so run the ingest step first.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from backend.agent import nodes
from backend.agent.graph import run_agent
from backend.agent.state import Message
from backend.config import settings


class FakeReply:
    """Minimal stand-in for a LangChain AIMessage."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.usage_metadata = {"input_tokens": 120, "output_tokens": 40}


class FakeLLM:
    """Scripted model whose reply depends on which node is calling it."""

    def __init__(self, temperature: float) -> None:
        self.temperature = temperature

    def invoke(self, messages: list[Any]) -> FakeReply:
        """Return a canned reply matched to the calling node's instructions."""
        system_text = str(getattr(messages[0], "content", ""))
        last_text = str(getattr(messages[-1], "content", "")).lower()

        if "Classify the customer's latest message" in system_text:
            return FakeReply(self._classify(last_text))
        if "Extract appointment details" in system_text:
            return FakeReply(json.dumps(self._extract_booking(last_text)))
        if "Knowledge base passages" in system_text:
            return FakeReply(
                "Based on what I have here, Saturday hours are 9:00 AM to 1:00 PM "
                "for urgent same-week visits."
            )
        return FakeReply("Sure - are you after some information, or would you like to book a visit?")

    @staticmethod
    def _classify(text: str) -> str:
        """Pick an intent the way a small model would from obvious keywords."""
        if any(word in text for word in ("idiot", "useless", "stupid", "human", "manager")):
            return "escalate"
        if any(word in text for word in ("appointment", "book", "schedule")):
            return "book_appointment"
        if "?" in text:
            return "faq"
        return "unknown"

    @staticmethod
    def _extract_booking(text: str) -> dict[str, str]:
        """Return only the booking fields the caller actually stated."""
        extracted = {"name": "", "date": "", "time": "", "reason": ""}
        if "maria chen" in text:
            extracted["name"] = "Maria Chen"
        if "2026-10-06" in text:
            extracted["date"] = "2026-10-06"
        if "2:00 pm" in text:
            extracted["time"] = "2:00 PM"
        if "blood pressure" in text:
            extracted["reason"] = "blood pressure follow-up"
        return extracted


@pytest.fixture(autouse=True)
def fake_llm_and_temp_files(monkeypatch, tmp_path):
    """Swap in the scripted model and write JSON records to a temp folder."""
    monkeypatch.setattr(nodes, "get_llm", lambda temperature: FakeLLM(temperature))
    monkeypatch.setattr(settings, "appointments_path", tmp_path / "appointments.json")
    monkeypatch.setattr(settings, "escalations_path", tmp_path / "escalations.json")
    yield


def chat(message: str, history: list[Message] | None = None) -> dict[str, Any]:
    """Run one turn through the real graph and return the final state."""
    messages = list(history or [])
    messages.append(Message(role="user", content=message))
    return asyncio.run(run_agent("test-session", message, messages))


def read_json(path) -> list[dict[str, Any]]:
    """Read a JSON array written by the tools, or [] when absent."""
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def test_faq_with_answer_in_knowledge_base_is_confident():
    """A question the documents cover is answered from retrieval."""
    state = chat("What are your Saturday hours?")

    assert state["intent"] == "faq"
    assert state["confidence"] >= settings.escalation_confidence_threshold
    assert state["action_taken"] == "answered_from_knowledge_base"
    assert not state["needs_escalation"]
    assert "9:00" in state["response"]


def test_faq_without_answer_escalates_instead_of_guessing():
    """An off-topic question falls below the threshold and hands off."""
    state = chat("Who won the 1994 world series?")

    assert state["confidence"] < settings.escalation_confidence_threshold
    assert state["needs_escalation"]
    assert state["action_taken"] == "escalate_to_human"
    assert read_json(settings.escalations_path), "escalation should be logged"


def test_booking_happy_path_returns_booking_id():
    """A message with every detail books the visit and confirms it."""
    state = chat(
        "I'd like to book an appointment for Maria Chen on 2026-10-06 at 2:00 PM "
        "for a blood pressure follow-up."
    )

    assert state["intent"] == "book_appointment"
    assert state["action_taken"] == "book_appointment"
    assert "APT-" in state["response"]

    saved = read_json(settings.appointments_path)
    assert len(saved) == 1
    assert saved[0]["name"] == "Maria Chen"
    assert saved[0]["date"] == "2026-10-06"


def test_booking_without_date_asks_for_it():
    """Missing details produce a clarifying question, not a guessed booking."""
    state = chat("I want to book an appointment for Maria Chen for a blood pressure follow-up.")

    assert state["action_taken"] == "requested_missing_booking_info"
    assert "date" in state["response"].lower()
    assert read_json(settings.appointments_path) == []


def test_offensive_input_is_deflected_gracefully():
    """Abuse is handed to a human rather than answered in kind."""
    state = chat("You people are useless idiots.")

    assert state["needs_escalation"]
    assert state["action_taken"] == "escalate_to_human"
    assert state["response"].strip()
    assert "idiot" not in state["response"].lower()
