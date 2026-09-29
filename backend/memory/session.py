"""In-memory conversation store, one record per session id.

Only the last few messages are sent to the LLM. Once a conversation runs
long, the older half is compressed into a summary so context stays small
without losing what was already discussed.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from backend.agent.state import Message
from backend.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """Everything remembered about one conversation."""

    session_id: str
    messages: list[Message] = field(default_factory=list)
    intent_history: list[str] = field(default_factory=list)
    actions_taken: list[str] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    booking_info: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    last_active_at: datetime = field(default_factory=datetime.now)

    @property
    def turns(self) -> int:
        """Number of user messages in this conversation."""
        return sum(1 for message in self.messages if message["role"] == "user")

    @property
    def average_confidence(self) -> float:
        """Mean retrieval confidence across turns that searched the knowledge base."""
        if not self.confidences:
            return 0.0
        return sum(self.confidences) / len(self.confidences)


def _summarize(messages: list[Message], previous_summary: str) -> str:
    """Compress older messages into a short plain-text summary.

    Uses the LLM when it is reachable and falls back to a truncated
    transcript so a provider outage cannot break the conversation.
    """
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from backend.agent.nodes import call_llm

        prompt = [
            SystemMessage(
                content=(
                    "Summarize this customer service conversation in at most four "
                    "sentences. Keep names, dates, times, booking IDs, and anything "
                    "still unresolved. Write plain sentences, no bullet points."
                )
            ),
            HumanMessage(
                content=(
                    f"Earlier summary: {previous_summary or 'none'}\n\n"
                    f"Conversation:\n{transcript}"
                )
            ),
        ]
        return call_llm(prompt, settings.intent_temperature, "summarize_session")
    except Exception as exc:
        logger.warning("Falling back to a truncated summary: %s", exc)
        combined = f"{previous_summary} {transcript}".strip()
        return combined[-1000:]


class SessionStore:
    """Thread-safe dictionary of sessions keyed by session id."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> Session:
        """Return the session for this id, creating it on first contact."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = Session(session_id=session_id)
                self._sessions[session_id] = session
                logger.info("Started session %s", session_id)
            return session

    def get(self, session_id: str) -> Session | None:
        """Return an existing session, or None when the id is unknown."""
        with self._lock:
            return self._sessions.get(session_id)

    def all_sessions(self) -> list[Session]:
        """Return every session, newest activity first."""
        with self._lock:
            return sorted(
                self._sessions.values(),
                key=lambda session: session.last_active_at,
                reverse=True,
            )

    def add_user_message(self, session_id: str, content: str) -> Session:
        """Record what the caller said and return the updated session."""
        session = self.get_or_create(session_id)
        with self._lock:
            session.messages.append(Message(role="user", content=content))
            session.last_active_at = datetime.now()
        return session

    def add_agent_turn(
        self,
        session_id: str,
        response: str,
        intent: str,
        action_taken: str,
        confidence: float,
        booking_info: dict[str, Any] | None = None,
    ) -> Session:
        """Record the agent's reply and the metadata the dashboard shows."""
        session = self.get_or_create(session_id)
        with self._lock:
            session.messages.append(Message(role="assistant", content=response))
            session.intent_history.append(intent)
            session.actions_taken.append(action_taken)
            if action_taken == "search_knowledge_base" or confidence > 0:
                session.confidences.append(confidence)
            if booking_info:
                session.booking_info.update(booking_info)
            session.last_active_at = datetime.now()
        self._maybe_compress(session)
        return session

    def context_window(self, session_id: str) -> list[Message]:
        """Return the last few messages, prefixed by the summary when one exists."""
        session = self.get_or_create(session_id)
        with self._lock:
            window = list(session.messages[-settings.context_window_messages :])
            summary = session.summary
        if summary:
            window.insert(
                0,
                Message(role="assistant", content=f"[Earlier conversation summary] {summary}"),
            )
        return window

    def _maybe_compress(self, session: Session) -> None:
        """Summarize and drop older messages once the session runs long."""
        with self._lock:
            long_enough = session.turns > settings.summarize_after_turns
            keep = settings.context_window_messages
            older = session.messages[:-keep] if long_enough else []
            recent = session.messages[-keep:] if long_enough else []
            previous_summary = session.summary
        if not older:
            return

        summary = _summarize(older, previous_summary)
        with self._lock:
            session.summary = summary
            session.messages = recent
        logger.info(
            "Compressed %d older message(s) for session %s",
            len(older),
            session.session_id,
        )

    def reset(self) -> None:
        """Drop every session. Used by tests."""
        with self._lock:
            self._sessions.clear()


session_store = SessionStore()
