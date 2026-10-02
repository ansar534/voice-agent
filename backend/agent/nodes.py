"""Graph node functions.

Every node takes the AgentState and returns ONLY the fields it changed.
Because ``messages`` has an operator.add reducer, a node returns just the
new messages it produced and LangGraph appends them to the history.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from groq import Groq

from ..config import (
    BUSINESS_NAME,
    CONFIDENCE_THRESHOLD,
    GROQ_API_KEY,
    GROQ_MODEL,
    INTENT_TEMPERATURE,
    MAX_HISTORY_TURNS,
    MAX_OUTPUT_TOKENS,
    RESPONSE_TEMPERATURE,
    SYSTEM_PROMPT,
    groq_key_is_set,
)
from .state import BOOKING_FIELDS, VALID_INTENTS, AgentState
from .tools import book_appointment, escalate_to_human, search_knowledge_base

logger = logging.getLogger(__name__)

# Stops the unknown -> clarify -> classify loop from running forever.
MAX_CLARIFY_ATTEMPTS = 1

LLM_UNAVAILABLE_MESSAGE = (
    "I'm having trouble reaching my system right now. "
    "Let me connect you with our staff."
)

_client: Groq | None = None


def get_client() -> Groq:
    """Return a cached Groq client, creating it on first use.

    Raises RuntimeError when no key is configured so callers can fall
    back instead of failing with a confusing auth error.
    """
    global _client
    if _client is None:
        if not groq_key_is_set():
            raise RuntimeError("GROQ_API_KEY is not set in .env")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def call_llm(messages: list[dict[str, str]], temperature: float, purpose: str) -> str:
    """Send a chat completion to Groq and return the reply text.

    Logs input tokens, output tokens, and latency for every call.
    Raises RuntimeError on any failure, which each node catches and turns
    into a safe fallback rather than a crash.
    """
    started = time.perf_counter()
    try:
        completion = get_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.error("LLM call [%s] failed after %.0f ms: %s", purpose, elapsed_ms, exc)
        raise RuntimeError(f"Groq call failed: {exc}") from exc

    elapsed_ms = (time.perf_counter() - started) * 1000
    usage = getattr(completion, "usage", None)
    logger.info(
        "LLM [%s] model=%s temp=%.1f input_tokens=%s output_tokens=%s latency_ms=%.0f",
        purpose,
        GROQ_MODEL,
        temperature,
        getattr(usage, "prompt_tokens", "?"),
        getattr(usage, "completion_tokens", "?"),
        elapsed_ms,
    )

    text = (completion.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("Groq returned an empty response.")
    return text


def _last_user_message(state: AgentState) -> str:
    """Return the most recent user message text, or an empty string."""
    for message in reversed(state.get("messages", []) or []):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _recent_history(state: AgentState) -> list[dict[str, str]]:
    """Return the last MAX_HISTORY_TURNS messages in Groq's format."""
    history = state.get("messages", []) or []
    return [
        {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
        for m in history[-MAX_HISTORY_TURNS:]
    ]


def _transcript(state: AgentState) -> str:
    """Flatten recent messages into a plain transcript for summaries."""
    return "\n".join(
        f"{m.get('role')}: {m.get('content')}" for m in _recent_history(state)
    )


def _keyword_intent(text: str) -> str:
    """Guess an intent from obvious keywords when the LLM is unreachable."""
    lowered = text.lower()
    if any(word in lowered for word in ("human", "manager", "agent", "emergency", "urgent")):
        return "escalate"
    if any(word in lowered for word in ("appointment", "book", "schedule", "reschedule", "cancel")):
        return "book_appointment"
    if "?" in lowered or any(
        word in lowered for word in ("what", "when", "where", "how", "do you", "are you")
    ):
        return "faq"
    return "unknown"


# ---------------------------------------------------------------------------
# NODE 1
# ---------------------------------------------------------------------------
def classify_intent(state: AgentState) -> dict[str, Any]:
    """Label the latest user message with one of the four intents."""
    user_message = _last_user_message(state)
    system_prompt = (
        "You are an intent classifier. Given the user message, classify it as exactly "
        "one of: faq, book_appointment, escalate, unknown.\n"
        "faq = questions about clinic info, hours, services, policies, insurance\n"
        "book_appointment = wants to schedule, reschedule, or cancel a visit\n"
        "escalate = angry, urgent medical need, or explicitly asks for human\n"
        "unknown = unclear, greeting, or out of scope\n"
        "Respond with ONLY the intent word, nothing else."
    )

    try:
        raw = call_llm(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            INTENT_TEMPERATURE,
            "classify_intent",
        ).lower()
        intent = next((name for name in VALID_INTENTS if name in raw), "unknown")
    except RuntimeError as exc:
        logger.warning("Intent classification falling back to keywords: %s", exc)
        intent = _keyword_intent(user_message)

    logger.info("Session %s intent=%s", state.get("session_id", "?"), intent)
    return {"intent": intent, "turn_count": int(state.get("turn_count", 0)) + 1}


# ---------------------------------------------------------------------------
# NODE 2
# ---------------------------------------------------------------------------
def retrieve_context(state: AgentState) -> dict[str, Any]:
    """Fetch supporting passages and score how well they match the question.

    When confidence falls under CONFIDENCE_THRESHOLD the intent is flipped
    to "escalate", so a weak match never reaches the answer generator.
    """
    query = _last_user_message(state)
    result = search_knowledge_base(query)
    confidence = float(result["confidence"])

    update: dict[str, Any] = {
        "retrieved_context": result["context"],
        "confidence": confidence,
    }
    if confidence < CONFIDENCE_THRESHOLD:
        logger.info(
            "Confidence %.3f below %.2f, routing to escalation",
            confidence,
            CONFIDENCE_THRESHOLD,
        )
        update["intent"] = "escalate"
        update["needs_escalation"] = True
    return update


# ---------------------------------------------------------------------------
# NODE 3
# ---------------------------------------------------------------------------
def generate_response(state: AgentState) -> dict[str, Any]:
    """Answer the question strictly from the retrieved context."""
    context = state.get("retrieved_context", "") or "No context available."
    prompt: list[dict[str, str]] = [
        {
            "role": "system",
            "content": f"{SYSTEM_PROMPT}\n\nContext from the knowledge base:\n{context}",
        }
    ]
    prompt.extend(_recent_history(state))

    try:
        answer = call_llm(prompt, RESPONSE_TEMPERATURE, "generate_response")
        action = "faq_response"
    except RuntimeError as exc:
        logger.error("Response generation failed: %s", exc)
        answer = LLM_UNAVAILABLE_MESSAGE
        action = "llm_unavailable"

    return {
        "messages": [{"role": "assistant", "content": answer}],
        "action_taken": action,
    }


# ---------------------------------------------------------------------------
# NODE 4
# ---------------------------------------------------------------------------
def collect_booking_info(state: AgentState) -> dict[str, Any]:
    """Pull booking details out of the conversation and ask for what is missing.

    Extraction is deliberately conservative: the model is told to leave a
    field blank rather than guess, so the agent never invents a date.
    """
    booking: dict[str, str] = dict(state.get("booking_info", {}) or {})
    extraction_prompt = [
        {
            "role": "system",
            "content": (
                "Extract appointment details from the conversation. Reply with JSON "
                'only, using exactly these keys: {"name": "", "date": "", "time": "", '
                '"reason": ""}. Use an empty string for anything the customer has not '
                "stated. Never guess a date, a time, or a name."
            ),
        },
        {
            "role": "user",
            "content": f"Conversation:\n{_transcript(state)}",
        },
    ]

    try:
        raw = call_llm(extraction_prompt, INTENT_TEMPERATURE, "collect_booking_info")
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        extracted = json.loads(match.group(0)) if match else {}
    except (RuntimeError, json.JSONDecodeError) as exc:
        logger.warning("Booking extraction failed: %s", exc)
        extracted = {}

    for field in BOOKING_FIELDS:
        value = str(extracted.get(field, "")).strip()
        if value:
            booking[field] = value

    missing = [field for field in BOOKING_FIELDS if not booking.get(field)]
    if not missing:
        return {"booking_info": booking, "action_taken": "booking_info_complete"}

    questions = {
        "name": "Of course. Can I get your full name?",
        "date": "What date would you like to come in?",
        "time": "What time works best for you?",
        "reason": "And what's the reason for your visit?",
    }
    return {
        "messages": [{"role": "assistant", "content": questions[missing[0]]}],
        "booking_info": booking,
        "action_taken": "collecting_booking_info",
    }


# ---------------------------------------------------------------------------
# NODE 5
# ---------------------------------------------------------------------------
def process_booking(state: AgentState) -> dict[str, Any]:
    """Save the appointment and confirm it to the caller."""
    booking = state.get("booking_info", {}) or {}
    confirmation = book_appointment(
        name=booking.get("name", ""),
        date=booking.get("date", ""),
        time=booking.get("time", ""),
        reason=booking.get("reason", ""),
    )
    return {
        "messages": [{"role": "assistant", "content": confirmation}],
        "action_taken": "appointment_booked",
        # Clear the slate so the next request starts fresh.
        "booking_info": {},
    }


# ---------------------------------------------------------------------------
# NODE 6
# ---------------------------------------------------------------------------
def handle_escalation(state: AgentState) -> dict[str, Any]:
    """Summarize the conversation, log the handoff, and tell the caller."""
    confidence = float(state.get("confidence", 0.0))
    if confidence and confidence < CONFIDENCE_THRESHOLD:
        reason = f"Retrieval confidence {confidence:.2f} below {CONFIDENCE_THRESHOLD}."
    else:
        reason = "Caller asked for a person or raised something staff must handle."

    summary_prompt = [
        {
            "role": "system",
            "content": (
                "Summarize this customer service conversation in exactly two sentences "
                "for the staff member taking over. State what the customer wants and "
                "anything still unresolved."
            ),
        },
        {"role": "user", "content": _transcript(state)},
    ]
    try:
        summary = call_llm(summary_prompt, INTENT_TEMPERATURE, "escalation_summary")
    except RuntimeError as exc:
        logger.warning("Summary generation failed, using raw transcript: %s", exc)
        summary = _transcript(state)[:500]

    handoff = escalate_to_human(reason=reason, conversation_summary=summary)
    return {
        "messages": [{"role": "assistant", "content": handoff}],
        "needs_escalation": True,
        "action_taken": "escalated",
    }


# ---------------------------------------------------------------------------
# NODE 7
# ---------------------------------------------------------------------------
def handle_unknown(state: AgentState) -> dict[str, Any]:
    """Ask the caller to say more about what they need.

    The graph loops back to classify_intent after this, which gives a
    transient misclassification one chance to resolve. The question is
    only added on the first pass so the retry cannot repeat it.
    """
    attempts = int(state.get("clarify_attempts", 0))
    update: dict[str, Any] = {
        "action_taken": "clarified",
        "clarify_attempts": attempts + 1,
    }
    if attempts == 0:
        update["messages"] = [
            {
                "role": "assistant",
                "content": (
                    f"I can help with {BUSINESS_NAME} information or booking an "
                    "appointment. Could you tell me more about what you need?"
                ),
            }
        ]
    return update


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def route_by_intent(state: AgentState) -> str:
    """Pick the branch matching the classified intent."""
    intent = state.get("intent", "unknown")
    return intent if intent in VALID_INTENTS else "unknown"


def route_after_booking_info(state: AgentState) -> str:
    """Book once every detail is known, otherwise return the question."""
    if state.get("action_taken") == "booking_info_complete":
        return "booking_info_complete"
    return "incomplete"


def route_after_unknown(state: AgentState) -> str:
    """Retry classification once, then stop and wait for the caller.

    The spec loops handle_unknown straight back to classify_intent. With
    no new user input that would re-classify the same text as unknown
    forever, so the retry is capped.
    """
    if int(state.get("clarify_attempts", 0)) <= MAX_CLARIFY_ATTEMPTS:
        return "retry"
    return "done"
