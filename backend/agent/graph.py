"""The compiled LangGraph state machine and the entry point that runs it.

    classify_intent
      faq              -> retrieve_context -> generate_response -> END
      book_appointment -> collect_booking_info -> process_booking -> END
                                               -> END  (still collecting)
      escalate         -> handle_escalation -> END
      unknown          -> handle_unknown -> classify_intent (one retry) -> END

retrieve_context flips the intent to "escalate" when confidence is too
low, so a weak match is handled by the next node rather than answered.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langgraph.graph import END, StateGraph

from ..memory.session import session_memory
from .nodes import (
    classify_intent,
    collect_booking_info,
    generate_response,
    handle_escalation,
    handle_unknown,
    process_booking,
    retrieve_context,
    route_after_booking_info,
    route_after_unknown,
    route_by_intent,
)
from .state import AgentState

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """Wire every node and edge into an uncompiled StateGraph."""
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("generate_response", generate_response)
    graph.add_node("collect_booking_info", collect_booking_info)
    graph.add_node("process_booking", process_booking)
    graph.add_node("handle_escalation", handle_escalation)
    graph.add_node("handle_unknown", handle_unknown)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "faq": "retrieve_context",
            "book_appointment": "collect_booking_info",
            "escalate": "handle_escalation",
            "unknown": "handle_unknown",
        },
    )

    # retrieve_context may downgrade the intent to escalate, so the next
    # hop is decided after it runs rather than hardwired.
    graph.add_conditional_edges(
        "retrieve_context",
        lambda state: "escalate" if state.get("intent") == "escalate" else "answer",
        {"answer": "generate_response", "escalate": "handle_escalation"},
    )

    graph.add_conditional_edges(
        "collect_booking_info",
        route_after_booking_info,
        {"booking_info_complete": "process_booking", "incomplete": END},
    )

    graph.add_conditional_edges(
        "handle_unknown",
        route_after_unknown,
        {"retry": "classify_intent", "done": END},
    )

    graph.add_edge("generate_response", END)
    graph.add_edge("process_booking", END)
    graph.add_edge("handle_escalation", END)

    return graph


agent_graph = build_graph().compile()


def run_agent(session_id: str, user_message: str) -> dict[str, Any]:
    """Run one user message through the agent and return the reply.

    Loads the session's recent history, invokes the graph, persists the
    new messages, and reports the intent, confidence, and action so the
    dashboard can show what happened.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    session = session_memory.get_session(session_id)
    history = session_memory.get_history(session_id)

    initial_state: AgentState = {
        "messages": [*history, {"role": "user", "content": user_message}],
        "session_id": session_id,
        "intent": "unknown",
        "retrieved_context": "",
        "confidence": 0.0,
        "needs_escalation": False,
        "action_taken": "none",
        # Carry partial booking details forward across turns.
        "booking_info": dict(session.get("booking_info", {})),
        "turn_count": int(session.get("turn_count", 0)),
        "clarify_attempts": 0,
    }

    try:
        final_state = agent_graph.invoke(initial_state)
    except Exception as exc:
        logger.exception("Agent run failed for session %s", session_id)
        fallback = (
            "Sorry, something went wrong on my end. "
            "Let me connect you with our staff."
        )
        session_memory.save_turn(
            session_id=session_id,
            user_message=user_message,
            assistant_message=fallback,
            intent="escalate",
            action="agent_error",
            confidence=0.0,
        )
        return {
            "response": fallback,
            "intent": "escalate",
            "confidence": 0.0,
            "action_taken": "agent_error",
            "session_id": session_id,
            "needs_escalation": True,
            "error": str(exc),
        }

    reply = ""
    for message in reversed(final_state.get("messages", [])):
        if message.get("role") == "assistant":
            reply = str(message.get("content", ""))
            break
    if not reply:
        reply = "I'm not sure I caught that. Could you tell me a bit more?"

    session_memory.save_turn(
        session_id=session_id,
        user_message=user_message,
        assistant_message=reply,
        intent=str(final_state.get("intent", "unknown")),
        action=str(final_state.get("action_taken", "none")),
        confidence=float(final_state.get("confidence", 0.0)),
        booking_info=final_state.get("booking_info", {}),
    )

    return {
        "response": reply,
        "intent": final_state.get("intent", "unknown"),
        "confidence": float(final_state.get("confidence", 0.0)),
        "action_taken": final_state.get("action_taken", "none"),
        "session_id": session_id,
        "needs_escalation": bool(final_state.get("needs_escalation", False)),
    }
