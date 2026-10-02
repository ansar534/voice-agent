"""In-memory conversation store, one entry per session id.

Phase 1 keeps everything in a process dictionary: restarting the API
clears all sessions. Appointments and escalations survive because the
tools write them to disk.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any

from ..config import MAX_HISTORY_TURNS

logger = logging.getLogger(__name__)


class SessionMemory:
    """Thread-safe store of per-session conversation history and stats."""

    def __init__(self) -> None:
        """Create an empty store."""
        self.sessions: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Return a session, creating it on first contact."""
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                session = {
                    "session_id": session_id,
                    "messages": [],
                    "intent_history": [],
                    "actions_taken": [],
                    "confidences": [],
                    "booking_info": {},
                    "start_time": datetime.now(),
                    "last_active": datetime.now(),
                    "turn_count": 0,
                }
                self.sessions[session_id] = session
                logger.info("Started session %s", session_id)
            return session

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """Return the last MAX_HISTORY_TURNS messages for a session.

        This sliding window is what gets replayed to the LLM, which keeps
        prompts small on long conversations.
        """
        session = self.get_session(session_id)
        with self._lock:
            return list(session["messages"][-MAX_HISTORY_TURNS:])

    def get_full_history(self, session_id: str) -> list[dict[str, str]]:
        """Return every message in a session, for the history endpoint."""
        session = self.get_session(session_id)
        with self._lock:
            return list(session["messages"])

    def save_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        intent: str,
        action: str,
        confidence: float = 0.0,
        booking_info: dict[str, Any] | None = None,
    ) -> None:
        """Record one complete exchange and its metadata."""
        session = self.get_session(session_id)
        with self._lock:
            session["messages"].append({"role": "user", "content": user_message})
            session["messages"].append({"role": "assistant", "content": assistant_message})
            session["intent_history"].append(intent)
            session["actions_taken"].append(action)
            if confidence > 0:
                session["confidences"].append(confidence)
            # Replace rather than merge: an empty dict after booking is a
            # deliberate reset of the collected details.
            session["booking_info"] = dict(booking_info or {})
            session["turn_count"] += 1
            session["last_active"] = datetime.now()

    def save_history(
        self,
        session_id: str,
        messages: list[dict[str, str]],
        intent: str,
        action: str,
    ) -> None:
        """Overwrite a session's messages and append its latest metadata.

        Kept for callers that manage the message list themselves.
        """
        session = self.get_session(session_id)
        with self._lock:
            session["messages"] = list(messages)
            session["intent_history"].append(intent)
            session["actions_taken"].append(action)
            session["turn_count"] += 1
            session["last_active"] = datetime.now()

    def get_analytics(self) -> dict[str, Any]:
        """Aggregate stats across every session for the dashboard."""
        with self._lock:
            sessions = list(self.sessions.values())

        intent_counts: dict[str, int] = {}
        action_counts: dict[str, int] = {}
        confidences: list[float] = []
        total_turns = 0
        escalated_sessions = 0

        for session in sessions:
            total_turns += session["turn_count"]
            confidences.extend(session["confidences"])
            for intent in session["intent_history"]:
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
            for action in session["actions_taken"]:
                action_counts[action] = action_counts.get(action, 0) + 1
            if "escalated" in session["actions_taken"]:
                escalated_sessions += 1

        session_count = len(sessions)
        top_intent = max(intent_counts, key=intent_counts.get) if intent_counts else "n/a"

        return {
            "total_sessions": session_count,
            "intent_counts": intent_counts,
            "action_counts": action_counts,
            "top_intent": top_intent,
            "avg_turns": round(total_turns / session_count, 2) if session_count else 0.0,
            "escalation_rate": round(escalated_sessions / session_count, 3) if session_count else 0.0,
            "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
            "total_messages": sum(len(session["messages"]) for session in sessions),
        }

    def clear_session(self, session_id: str) -> bool:
        """Delete one session. Returns False when the id is unknown."""
        with self._lock:
            return self.sessions.pop(session_id, None) is not None

    def clear_all(self) -> None:
        """Delete every session. Used by tests."""
        with self._lock:
            self.sessions.clear()


session_memory = SessionMemory()
