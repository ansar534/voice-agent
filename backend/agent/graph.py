"""The LangGraph state machine that handles one user turn.

    START -> classify_intent
      faq              -> retrieve_context -> generate_response -> END
                                           -> escalate -> END   (low confidence)
      book_appointment -> collect_booking_info -> book_appointment -> END
                                               -> END            (missing details)
      escalate         -> escalate -> END
      unknown          -> clarify -> classify_intent (once) -> END
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from backend.agent import nodes
from backend.agent.state import AgentState, Message, new_state

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """Wire every node and edge into an uncompiled StateGraph."""
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", nodes.classify_intent)
    graph.add_node("retrieve_context", nodes.retrieve_context)
    graph.add_node("generate_response", nodes.generate_response)
    graph.add_node("collect_booking_info", nodes.collect_booking_info)
    graph.add_node("book_appointment", nodes.book_appointment_node)
    graph.add_node("escalate", nodes.escalate)
    graph.add_node("clarify", nodes.clarify)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        nodes.route_after_intent,
        {
            "faq": "retrieve_context",
            "book_appointment": "collect_booking_info",
            "escalate": "escalate",
            "unknown": "clarify",
        },
    )
    graph.add_conditional_edges(
        "retrieve_context",
        nodes.route_after_retrieval,
        {"generate_response": "generate_response", "escalate": "escalate"},
    )
    graph.add_conditional_edges(
        "collect_booking_info",
        nodes.route_after_booking_info,
        {"book_appointment": "book_appointment", "end": END},
    )
    graph.add_conditional_edges(
        "clarify",
        nodes.route_after_clarify,
        {"classify_intent": "classify_intent", "end": END},
    )
    graph.add_edge("generate_response", END)
    graph.add_edge("book_appointment", END)
    graph.add_edge("escalate", END)
    return graph


@lru_cache(maxsize=1)
def get_compiled_graph() -> Any:
    """Compile the graph once and reuse it for every request."""
    compiled = build_graph().compile()
    logger.info("Agent graph compiled")
    return compiled


async def run_agent(
    session_id: str,
    user_input: str,
    messages: list[Message],
    booking_info: dict[str, Any] | None = None,
) -> AgentState:
    """Run one user turn through the graph and return the final state.

    ``messages`` is the session's sliding window and must already include
    the current user message. ``booking_info`` carries appointment details
    gathered on earlier turns so a caller is not asked twice.
    """
    state = new_state(session_id, user_input, messages)
    if booking_info:
        state["booking_info"] = dict(booking_info)

    try:
        result = await get_compiled_graph().ainvoke(state)
    except Exception as exc:
        logger.exception("Agent run failed for session %s", session_id)
        state["response"] = (
            "Sorry, something went wrong on my end. "
            "Let me have a staff member follow up with you."
        )
        state["needs_escalation"] = True
        state["action_taken"] = "agent_error"
        state["error"] = str(exc)
        return state

    if not result.get("response"):
        result["response"] = (
            "I am not sure I caught that. Could you tell me a little more?"
        )
        result["action_taken"] = result.get("action_taken") or "no_response"
    return result
