"""Aggregate chat, call, retrieval, and cost metrics for the dashboard."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any

from .agent.tools import get_appointments, get_escalations
from .automation.post_call import get_call_analytics_records, get_support_tickets
from .config import (
    CONFIDENCE_THRESHOLD,
    COST_PER_DEEPGRAM_MINUTE,
    COST_PER_ELEVENLABS_1K_CHARS,
    COST_PER_GROQ_1K_TOKENS,
    COST_PER_HUMAN_AGENT_CALL,
    COST_PER_TWILIO_MINUTE,
)
from .memory.session import session_memory
from .phone.twilio_handler import get_calls


def build_analytics() -> dict[str, Any]:
    """Return the full analytics payload used by /analytics and the UI."""
    chat = session_memory.get_analytics()
    appointments = get_appointments()
    escalations = get_escalations()
    calls = get_calls()
    call_metrics = get_call_analytics_records()
    tickets = get_support_tickets()

    completed_calls = [c for c in calls if c.get("status") not in {"ringing", "test"}]
    missed_calls = [c for c in calls if c.get("status") in {"missed", "no-answer", "busy"}]
    resolved_calls = [
        c
        for c in completed_calls
        if c.get("status") in {"completed", "resolved"} or c.get("resolved")
    ]
    escalated_calls = [
        c for c in completed_calls if c.get("status") == "escalated" or "escalat" in str(c.get("status", ""))
    ]

    durations = [int(c.get("duration") or 0) for c in completed_calls if int(c.get("duration") or 0) > 0]
    avg_call_duration = round(sum(durations) / len(durations), 1) if durations else 0.0

    intent_counts = dict(chat.get("intent_counts") or {})
    intent_total = sum(intent_counts.values()) or 1
    intent_breakdown = {
        key: round(100.0 * intent_counts.get(key, 0) / intent_total, 1)
        for key in ("faq", "book_appointment", "escalate", "unknown")
    }

    booking_actions = sum(
        1 for action, count in (chat.get("action_counts") or {}).items() if action == "appointment_booked" for _ in range(count)
    )
    # Prefer action_counts map value when present.
    action_counts = chat.get("action_counts") or {}
    booking_actions = int(action_counts.get("appointment_booked", 0))
    total_sessions = int(chat.get("total_sessions") or 0)
    booking_success_rate = (
        round(booking_actions / total_sessions, 3) if total_sessions else 0.0
    )

    confidences = []
    low_confidence_queries: list[dict[str, Any]] = []
    faq_questions: list[str] = []
    for session in session_memory.sessions.values():
        confidences.extend(session.get("confidences") or [])
        messages = session.get("messages") or []
        intents = session.get("intent_history") or []
        for index, intent in enumerate(intents):
            user_msgs = [m for m in messages if m.get("role") == "user"]
            user_text = user_msgs[index]["content"] if index < len(user_msgs) else ""
            if intent == "faq" and user_text:
                faq_questions.append(str(user_text))
            conf = 0.0
            if index < len(session.get("confidences") or []):
                conf = float(session["confidences"][index])
            if conf and conf < CONFIDENCE_THRESHOLD and user_text:
                low_confidence_queries.append(
                    {"query": user_text, "confidence": conf, "session_id": session.get("session_id")}
                )

    for row in call_metrics:
        if row.get("confidence"):
            confidences.append(float(row["confidence"]))

    avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else float(
        chat.get("avg_confidence") or 0.0
    )

    # Hallucination proxy: escalations after an faq_response in the same session.
    hallucination_incidents = 0
    for session in session_memory.sessions.values():
        actions = session.get("actions_taken") or []
        if "faq_response" in actions and "escalated" in actions:
            hallucination_incidents += 1

    # Call trend last 7 days.
    today = datetime.now().date()
    trend_counts: dict[str, int] = defaultdict(int)
    for call in calls:
        raw = str(call.get("timestamp") or "")
        try:
            day = datetime.fromisoformat(raw).date()
        except ValueError:
            continue
        if today - timedelta(days=6) <= day <= today:
            trend_counts[day.isoformat()] += 1
    call_trend = [
        {"date": (today - timedelta(days=offset)).isoformat(), "calls": trend_counts[(today - timedelta(days=offset)).isoformat()]}
        for offset in range(6, -1, -1)
    ]

    top_faqs = [question for question, _ in Counter(faq_questions).most_common(10)]

    # Cost model (estimates — useful for the financial tab).
    total_call_minutes = max(sum(durations) / 60.0, len(completed_calls) * 1.5)
    chat_sessions = total_sessions
    estimated_groq_tokens = (chat.get("total_messages") or 0) * 350
    groq_cost = (estimated_groq_tokens / 1000.0) * COST_PER_GROQ_1K_TOKENS
    deepgram_cost = total_call_minutes * COST_PER_DEEPGRAM_MINUTE
    elevenlabs_cost = (chat_sessions * 120 / 1000.0) * COST_PER_ELEVENLABS_1K_CHARS
    twilio_cost = total_call_minutes * COST_PER_TWILIO_MINUTE
    ai_costs = groq_cost + deepgram_cost + elevenlabs_cost + twilio_cost
    revenue_savings = max(0.0, len(completed_calls) * COST_PER_HUMAN_AGENT_CALL - ai_costs)

    resolution_by_intent: dict[str, float] = {}
    for intent_name in ("faq", "book_appointment", "escalate", "unknown"):
        matching = [
            s
            for s in session_memory.sessions.values()
            if intent_name in (s.get("intent_history") or [])
        ]
        if not matching:
            resolution_by_intent[intent_name] = 0.0
            continue
        resolved = sum(
            1
            for s in matching
            if "escalated" not in (s.get("actions_taken") or [])
            and s.get("turn_count", 0) > 0
        )
        resolution_by_intent[intent_name] = round(100.0 * resolved / len(matching), 1)

    escalation_reasons = Counter(
        str(e.get("reason") or "unspecified")[:80] for e in escalations
    )

    # Mock CSAT until real follow-up surveys exist.
    csat_scores = []
    for session in session_memory.sessions.values():
        if not session.get("turn_count"):
            continue
        if "escalated" in (session.get("actions_taken") or []):
            csat_scores.append(3)
        elif "appointment_booked" in (session.get("actions_taken") or []):
            csat_scores.append(5)
        else:
            csat_scores.append(4)
    avg_csat = round(sum(csat_scores) / len(csat_scores), 2) if csat_scores else 0.0

    unanswered = [row["query"] for row in low_confidence_queries[:20]]

    return {
        # Chat (Phase 1 fields preserved)
        "total_sessions": total_sessions,
        "intent_counts": intent_counts,
        "action_counts": action_counts,
        "top_intent": chat.get("top_intent", "n/a"),
        "avg_turns": chat.get("avg_turns", 0.0),
        "escalation_rate": chat.get("escalation_rate", 0.0),
        "avg_confidence": avg_confidence,
        "total_messages": chat.get("total_messages", 0),
        "total_appointments": len(appointments),
        "total_escalations": len(escalations),
        # Call metrics
        "total_calls": len(completed_calls),
        "missed_calls": len(missed_calls),
        "avg_call_duration": avg_call_duration,
        "calls_resolved": len(resolved_calls),
        "calls_escalated": len(escalated_calls),
        "call_trend": call_trend,
        # Chat extras
        "unique_users": total_sessions,
        "booking_success_rate": booking_success_rate,
        "intent_breakdown": intent_breakdown,
        "top_faqs": top_faqs,
        # Retrieval quality
        "retrieval_avg_confidence": avg_confidence,
        "low_confidence_queries": low_confidence_queries[:20],
        "hallucination_incidents": hallucination_incidents,
        "unanswered_questions": unanswered,
        # Agent performance
        "resolution_rate_by_intent": resolution_by_intent,
        "escalation_reasons": dict(escalation_reasons),
        "avg_time_to_resolution_sec": round(float(chat.get("avg_turns", 0.0)) * 45.0, 1),
        "customer_satisfaction": avg_csat,
        "open_tickets": sum(1 for t in tickets if t.get("status") == "open"),
        # Financial
        "api_costs": {
            "groq": round(groq_cost, 4),
            "deepgram": round(deepgram_cost, 4),
            "elevenlabs": round(elevenlabs_cost, 4),
            "twilio": round(twilio_cost, 4),
            "total": round(ai_costs, 4),
        },
        "cost_per_call": round(ai_costs / len(completed_calls), 4) if completed_calls else 0.0,
        "cost_per_chat_session": round(ai_costs / chat_sessions, 4) if chat_sessions else 0.0,
        "revenue_savings": round(revenue_savings, 2),
        "calls_handled": len(completed_calls),
        # Call quality placeholders grounded in available signals
        "call_quality": {
            "clarity": round(min(1.0, avg_confidence + 0.15), 2) if avg_confidence else 0.0,
            "transcription_accuracy": round(min(1.0, avg_confidence + 0.1), 2) if avg_confidence else 0.0,
            "resolution_rate": (
                round(len(resolved_calls) / len(completed_calls), 3) if completed_calls else 0.0
            ),
        },
    }
