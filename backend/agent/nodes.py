"""Node functions that make up the agent graph.

Every node takes the current AgentState and returns only the fields it
changed. LLM calls go through call_llm, which logs token counts and
latency and converts provider failures into readable errors.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.agent.state import VALID_INTENTS, AgentState, Intent, Message
from backend.agent.tools import book_appointment, escalate_to_human
from backend.config import settings
from backend.retrieval.vectorstore import hybrid_search

logger = logging.getLogger(__name__)

MAX_CLARIFICATION_ATTEMPTS = 1
BOOKING_FIELDS = ("name", "date", "time", "reason")

_llm_cache: dict[float, Any] = {}


class LLMError(RuntimeError):
    """Raised when the Groq call fails or returns nothing usable."""


def get_llm(temperature: float) -> Any:
    """Return a cached ChatGroq client for the given temperature.

    Tests replace this function to run the graph without network access.
    """
    if temperature in _llm_cache:
        return _llm_cache[temperature]
    if not settings.groq_api_key or settings.groq_api_key == "your_groq_api_key_here":
        raise LLMError(
            "GROQ_API_KEY is not set. Add a real key to .env before chatting."
        )
    try:
        from langchain_groq import ChatGroq

        client = ChatGroq(
            model=settings.groq_model,
            temperature=temperature,
            max_tokens=settings.max_output_tokens,
            api_key=settings.groq_api_key,
        )
    except Exception as exc:
        raise LLMError(f"Could not create the Groq client: {exc}") from exc
    _llm_cache[temperature] = client
    return client


def call_llm(prompt_messages: list[Any], temperature: float, purpose: str) -> str:
    """Send messages to Groq and return the reply text.

    Logs input tokens, output tokens, and latency in milliseconds for
    every call. Raises LLMError so callers can degrade gracefully.
    """
    started = time.perf_counter()
    try:
        client = get_llm(temperature)
        reply = client.invoke(prompt_messages)
    except LLMError:
        raise
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.error("LLM call [%s] failed after %.0f ms: %s", purpose, elapsed_ms, exc)
        raise LLMError(f"The language model call failed: {exc}") from exc

    elapsed_ms = (time.perf_counter() - started) * 1000
    usage = getattr(reply, "usage_metadata", None) or {}
    logger.info(
        "LLM call [%s] model=%s temp=%.1f input_tokens=%s output_tokens=%s latency_ms=%.0f",
        purpose,
        settings.groq_model,
        temperature,
        usage.get("input_tokens", "unknown"),
        usage.get("output_tokens", "unknown"),
        elapsed_ms,
    )
    text = getattr(reply, "content", "")
    if isinstance(text, list):
        text = " ".join(str(part) for part in text)
    text = str(text).strip()
    if not text:
        raise LLMError("The language model returned an empty response.")
    return text


def _history_messages(state: AgentState) -> list[Any]:
    """Convert the session's sliding window into LangChain messages."""
    history: list[Any] = []
    stored: list[Message] = state.get("messages", []) or []
    for message in stored[-settings.context_window_messages :]:
        if message["role"] == "user":
            history.append(HumanMessage(content=message["content"]))
        else:
            history.append(AIMessage(content=message["content"]))
    return history


def _extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of an LLM reply.

    Models often wrap JSON in prose or code fences, so the braces are
    located rather than parsing the whole string.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _fallback_intent(user_input: str) -> Intent:
    """Guess an intent from keywords when the LLM is unavailable."""
    text = user_input.lower()
    if any(word in text for word in ("human", "agent", "manager", "representative", "person")):
        return "escalate"
    if any(word in text for word in ("appointment", "book", "schedule", "reschedule", "slot")):
        return "book_appointment"
    if "?" in text or any(word in text for word in ("what", "when", "where", "how", "do you", "are you")):
        return "faq"
    return "unknown"


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------


def classify_intent(state: AgentState) -> dict[str, Any]:
    """Label the current message as faq, book_appointment, escalate, or unknown."""
    user_input = state.get("user_input", "")
    instructions = (
        "Classify the customer's latest message into exactly one intent.\n"
        "faq: a question about hours, location, services, policies, insurance, or how things work.\n"
        "book_appointment: wants to book, change, or cancel an appointment, or asks about open times.\n"
        "escalate: asks for a human, is angry, has an emergency, or needs something staff must handle.\n"
        "unknown: greetings, chit-chat, or anything too vague to act on.\n"
        "Reply with the single intent word and nothing else."
    )
    prompt = [SystemMessage(content=instructions), *_history_messages(state)]
    prompt.append(HumanMessage(content=f"Latest message: {user_input}"))

    try:
        raw = call_llm(prompt, settings.intent_temperature, "classify_intent").lower()
    except LLMError as exc:
        intent = _fallback_intent(user_input)
        logger.warning("Intent classification fell back to keywords: %s", exc)
        return {"intent": intent, "error": str(exc)}

    intent: Intent = "unknown"
    for candidate in VALID_INTENTS:
        if candidate in raw:
            intent = candidate
            break
    logger.info("Session %s classified as '%s'", state.get("session_id", "?"), intent)
    return {"intent": intent, "error": ""}


def retrieve_context(state: AgentState) -> dict[str, Any]:
    """Run hybrid retrieval for the current question and record confidence."""
    query = state.get("user_input", "")
    try:
        result = hybrid_search(query, top_k=settings.retrieval_top_k)
        chunks = [
            {
                "source": chunk.source,
                "text": chunk.text,
                "similarity": round(chunk.cosine_similarity, 4),
            }
            for chunk in result.chunks
        ]
        confidence = result.confidence
    except Exception as exc:
        logger.error("Retrieval failed for %r: %s", query, exc)
        return {
            "retrieved_context": [],
            "confidence": 0.0,
            "needs_escalation": True,
            "action_taken": "retrieval_failed",
            "error": f"Knowledge base lookup failed: {exc}",
        }

    needs_escalation = confidence < settings.escalation_confidence_threshold
    return {
        "retrieved_context": chunks,
        "confidence": confidence,
        "needs_escalation": needs_escalation,
        "action_taken": "search_knowledge_base",
    }


def generate_response(state: AgentState) -> dict[str, Any]:
    """Answer the question using only the retrieved passages."""
    chunks = state.get("retrieved_context", []) or []
    context_text = "\n\n".join(
        f"(source: {chunk['source']})\n{chunk['text']}" for chunk in chunks
    )
    instructions = (
        f"{settings.system_prompt}\n\n"
        "Answer using ONLY the knowledge base passages below. "
        "Never invent details that are not in them. If the passages do not "
        "cover the question, say plainly that you do not have that information "
        "and offer to have a staff member follow up. "
        "Keep the answer to two or three sentences, the way you would say it on a call.\n\n"
        f"Knowledge base passages:\n{context_text}"
    )
    prompt = [SystemMessage(content=instructions), *_history_messages(state)]
    prompt.append(HumanMessage(content=state.get("user_input", "")))

    try:
        answer = call_llm(prompt, settings.response_temperature, "generate_response")
    except LLMError as exc:
        return {
            "response": (
                "I am having trouble reaching my system right now. "
                "Let me have a staff member follow up with you."
            ),
            "needs_escalation": True,
            "action_taken": "llm_unavailable",
            "error": str(exc),
        }
    return {"response": answer, "action_taken": "answered_from_knowledge_base"}


def collect_booking_info(state: AgentState) -> dict[str, Any]:
    """Pull name, date, time, and reason out of the conversation.

    Missing fields produce a clarifying question instead of a booking.
    """
    known: dict[str, Any] = dict(state.get("booking_info", {}) or {})
    transcript = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in (state.get("messages", []) or [])[-settings.context_window_messages :]
    )
    instructions = (
        "Extract appointment details from the conversation. "
        'Reply with JSON only, using exactly these keys: {"name": "", "date": "", '
        '"time": "", "reason": ""}. '
        "Use an empty string for anything the customer has not stated. "
        "Dates must be YYYY-MM-DD. Never guess a date or a time."
    )
    prompt = [
        SystemMessage(content=instructions),
        HumanMessage(content=f"Conversation:\n{transcript}\n\nLatest: {state.get('user_input', '')}"),
    ]

    try:
        raw = call_llm(prompt, settings.intent_temperature, "collect_booking_info")
        extracted = _extract_json_object(raw)
    except LLMError as exc:
        logger.warning("Booking extraction failed: %s", exc)
        extracted = {}

    for field in BOOKING_FIELDS:
        value = str(extracted.get(field, "")).strip()
        if value:
            known[field] = value

    missing = [field for field in BOOKING_FIELDS if not known.get(field)]
    if missing:
        readable = {
            "name": "your full name",
            "date": "the date you would like",
            "time": "the time that works for you",
            "reason": "the reason for the visit",
        }
        labels = [readable[field] for field in missing]
        if len(labels) == 1:
            wanted = labels[0]
        else:
            wanted = f"{', '.join(labels[:-1])}, and {labels[-1]}"
        return {
            "booking_info": known,
            "response": f"Happy to get that booked. Could you give me {wanted}?",
            "action_taken": "requested_missing_booking_info",
        }
    return {"booking_info": known, "action_taken": "collected_booking_info"}


def book_appointment_node(state: AgentState) -> dict[str, Any]:
    """Call the booking tool and return the confirmation to the caller."""
    booking = state.get("booking_info", {}) or {}
    try:
        confirmation = book_appointment.invoke(
            {
                "name": booking.get("name", ""),
                "date": booking.get("date", ""),
                "time": booking.get("time", ""),
                "reason": booking.get("reason", ""),
            }
        )
    except Exception as exc:
        logger.error("Booking tool failed: %s", exc)
        return {
            "response": "I could not save that appointment. Let me get a staff member to help.",
            "needs_escalation": True,
            "action_taken": "book_appointment_failed",
            "error": str(exc),
        }
    return {"response": confirmation, "action_taken": "book_appointment"}


def escalate(state: AgentState) -> dict[str, Any]:
    """Log the handoff and tell the caller a person will follow up."""
    confidence = state.get("confidence", 0.0)
    if state.get("intent") == "escalate":
        reason = "Caller asked for a human or raised something staff must handle."
    elif confidence < settings.escalation_confidence_threshold:
        reason = f"Retrieval confidence {confidence:.2f} below threshold."
    else:
        reason = "Agent could not complete the request."

    summary = " | ".join(
        f"{message['role']}: {message['content']}"
        for message in (state.get("messages", []) or [])[-settings.context_window_messages :]
    )
    try:
        handoff = escalate_to_human.invoke({"reason": reason, "summary": summary})
    except Exception as exc:
        logger.error("Escalation tool failed: %s", exc)
        handoff = "Let me pass this to one of our staff members, who will follow up shortly."
    return {
        "response": handoff,
        "needs_escalation": True,
        "action_taken": "escalate_to_human",
    }


def clarify(state: AgentState) -> dict[str, Any]:
    """Ask one short question to find out what the caller needs."""
    attempts = int(state.get("clarification_attempts", 0)) + 1
    instructions = (
        f"{settings.system_prompt}\n\n"
        "You could not tell what the customer needs. Reply with one short, friendly "
        "question that finds out whether they want information or an appointment. "
        "Do not answer anything else."
    )
    prompt = [SystemMessage(content=instructions), *_history_messages(state)]
    prompt.append(HumanMessage(content=state.get("user_input", "")))

    try:
        question = call_llm(prompt, settings.response_temperature, "clarify")
    except LLMError as exc:
        logger.warning("Clarification fell back to a fixed prompt: %s", exc)
        question = (
            "Happy to help. Are you looking for information about the clinic, "
            "or would you like to book an appointment?"
        )
    return {
        "response": question,
        "clarification_attempts": attempts,
        "action_taken": "clarify",
    }


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------


def route_after_intent(state: AgentState) -> str:
    """Send the turn down the branch matching the classified intent."""
    intent = state.get("intent", "unknown")
    if intent in ("faq", "book_appointment", "escalate"):
        return intent
    if int(state.get("clarification_attempts", 0)) >= MAX_CLARIFICATION_ATTEMPTS:
        # The caller was already asked once, so stop looping and hand off.
        return "escalate"
    return "unknown"


def route_after_retrieval(state: AgentState) -> str:
    """Escalate weak retrieval instead of letting the model guess."""
    if state.get("needs_escalation"):
        logger.info(
            "Confidence %.2f below %.2f, escalating",
            state.get("confidence", 0.0),
            settings.escalation_confidence_threshold,
        )
        return "escalate"
    return "generate_response"


def route_after_booking_info(state: AgentState) -> str:
    """Book only once every required detail is known."""
    booking = state.get("booking_info", {}) or {}
    if all(booking.get(field) for field in BOOKING_FIELDS):
        return "book_appointment"
    return "end"


def route_after_clarify(state: AgentState) -> str:
    """Re-classify once after clarifying, then stop to wait for the caller."""
    if int(state.get("clarification_attempts", 0)) < MAX_CLARIFICATION_ATTEMPTS:
        return "classify_intent"
    return "end"
