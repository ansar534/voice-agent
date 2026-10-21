"""Advanced analytics dashboard spanning chat, calls, retrieval, and cost."""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests
import streamlit as st

API_BASE = "http://localhost:8000"
REQUEST_TIMEOUT = 60


def _get(path: str) -> Any | None:
    try:
        response = requests.get(f"{API_BASE}{path}", timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach the API at {API_BASE}.")
    except requests.exceptions.RequestException as exc:
        st.error(f"Request to {path} failed: {exc}")
    return None


def _post(path: str, payload: dict | None = None) -> Any | None:
    try:
        response = requests.post(
            f"{API_BASE}{path}", json=payload or {}, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        st.error(f"Request to {path} failed: {exc}")
    return None


def show_analytics_dashboard() -> None:
    """Render the multi-tab advanced analytics page."""
    st.markdown("## Advanced Analytics")

    header, refresh = st.columns([5, 1])
    with refresh:
        if st.button("Refresh", use_container_width=True, key="analytics_refresh"):
            st.rerun()

    stats = _get("/analytics")
    if stats is None:
        return

    (
        calls_tab,
        chat_tab,
        retrieval_tab,
        performance_tab,
        financial_tab,
    ) = st.tabs(["Calls", "Chat", "Retrieval Quality", "Agent Performance", "Financial"])

    with calls_tab:
        _render_calls(stats)
    with chat_tab:
        _render_chat(stats)
    with retrieval_tab:
        _render_retrieval(stats)
    with performance_tab:
        _render_performance(stats)
    with financial_tab:
        _render_financial(stats)

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Re-ingest Knowledge Base"):
            with st.spinner("Rebuilding indexes..."):
                result = _post("/ingest")
            if result:
                st.success(
                    f"Indexed {result['chunks_stored']} chunks from "
                    f"{result['documents']} documents."
                )
    with col_b:
        st.markdown(
            f"[Download call transcripts (CSV)]({API_BASE}/analytics/transcripts.csv)"
        )


def _render_calls(stats: dict[str, Any]) -> None:
    one, two, three, four = st.columns(4)
    one.metric("Total calls", stats.get("total_calls", 0))
    two.metric("Missed calls", stats.get("missed_calls", 0))
    three.metric("Avg duration (s)", stats.get("avg_call_duration", 0))
    four.metric("Calls resolved", stats.get("calls_resolved", 0))

    quality = stats.get("call_quality") or {}
    q1, q2, q3 = st.columns(3)
    q1.metric("Clarity", f"{float(quality.get('clarity', 0)) * 100:.0f}%")
    q2.metric(
        "Transcription accuracy",
        f"{float(quality.get('transcription_accuracy', 0)) * 100:.0f}%",
    )
    q3.metric(
        "Resolution rate",
        f"{float(quality.get('resolution_rate', 0)) * 100:.0f}%",
    )

    st.markdown("### Call trend (last 7 days)")
    trend = stats.get("call_trend") or []
    if trend:
        frame = pd.DataFrame(trend).set_index("date")
        st.bar_chart(frame)
    else:
        st.caption("No call activity yet.")

    st.markdown("### Call history")
    calls = _get("/calls")
    if calls:
        st.dataframe(pd.DataFrame(calls), use_container_width=True, hide_index=True)
    elif calls is not None:
        st.caption("No phone calls logged yet.")


def _render_chat(stats: dict[str, Any]) -> None:
    one, two, three, four = st.columns(4)
    one.metric("Total sessions", stats.get("total_sessions", 0))
    two.metric("Unique users", stats.get("unique_users", 0))
    three.metric("Avg turns", stats.get("avg_turns", 0))
    four.metric("Booking success", f"{float(stats.get('booking_success_rate', 0)) * 100:.0f}%")

    st.markdown("### Intent distribution")
    breakdown = stats.get("intent_breakdown") or {}
    intent_counts = stats.get("intent_counts") or {}
    if intent_counts:
        st.bar_chart(pd.DataFrame({"count": intent_counts}))
        st.caption(
            " · ".join(f"{name}: {pct}%" for name, pct in breakdown.items())
        )
    else:
        st.caption("No conversations yet.")

    st.markdown("### Top frequently asked questions")
    faqs = stats.get("top_faqs") or []
    if faqs:
        st.dataframe(
            pd.DataFrame({"question": faqs}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No FAQ questions captured yet.")

    st.markdown("### Appointments")
    appointments = _get("/appointments")
    if appointments:
        st.dataframe(pd.DataFrame(appointments), use_container_width=True, hide_index=True)
    elif appointments is not None:
        st.caption("No appointments booked yet.")


def _render_retrieval(stats: dict[str, Any]) -> None:
    avg_conf = float(stats.get("retrieval_avg_confidence") or stats.get("avg_confidence") or 0)
    one, two, three = st.columns(3)
    one.metric("Avg confidence", f"{avg_conf:.2f}")
    two.metric("Hallucination incidents", stats.get("hallucination_incidents", 0))
    three.metric(
        "Target",
        "≥ 0.70",
        delta="on track" if avg_conf >= 0.7 else "below target",
        delta_color="normal" if avg_conf >= 0.7 else "inverse",
    )

    st.markdown("### Low-confidence queries escalated (< 0.4)")
    low = stats.get("low_confidence_queries") or []
    if low:
        st.dataframe(pd.DataFrame(low), use_container_width=True, hide_index=True)
    else:
        st.caption("No low-confidence escalations yet.")

    st.markdown("### Knowledge base coverage gaps")
    unanswered = stats.get("unanswered_questions") or []
    if unanswered:
        st.dataframe(
            pd.DataFrame({"unanswered": unanswered}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No coverage gaps recorded.")


def _render_performance(stats: dict[str, Any]) -> None:
    one, two, three = st.columns(3)
    one.metric("Escalation rate", f"{float(stats.get('escalation_rate', 0)) * 100:.0f}%")
    two.metric(
        "Avg time to resolution (s)",
        stats.get("avg_time_to_resolution_sec", 0),
    )
    three.metric("Customer satisfaction", f"{float(stats.get('customer_satisfaction', 0)):.1f} / 5")

    st.markdown("### Resolution rate by intent")
    resolution = stats.get("resolution_rate_by_intent") or {}
    if resolution:
        st.bar_chart(pd.DataFrame({"resolution_%": resolution}))
    else:
        st.caption("No resolution data yet.")

    st.markdown("### Escalation rate by reason")
    reasons = stats.get("escalation_reasons") or {}
    if reasons:
        st.dataframe(
            pd.DataFrame(
                [{"reason": reason, "count": count} for reason, count in reasons.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No escalations logged yet.")

    st.metric("Open support tickets", stats.get("open_tickets", 0))


def _render_financial(stats: dict[str, Any]) -> None:
    costs = stats.get("api_costs") or {}
    one, two, three, four = st.columns(4)
    one.metric("Groq", f"${float(costs.get('groq', 0)):.4f}")
    two.metric("Deepgram", f"${float(costs.get('deepgram', 0)):.4f}")
    three.metric("ElevenLabs", f"${float(costs.get('elevenlabs', 0)):.4f}")
    four.metric("Twilio", f"${float(costs.get('twilio', 0)):.4f}")

    st.metric("Total AI costs", f"${float(costs.get('total', 0)):.4f}")

    a, b, c = st.columns(3)
    a.metric("Cost per call", f"${float(stats.get('cost_per_call', 0)):.4f}")
    b.metric("Cost per chat session", f"${float(stats.get('cost_per_chat_session', 0)):.4f}")
    c.metric("Revenue savings", f"${float(stats.get('revenue_savings', 0)):.2f}")

    st.caption(
        f"Savings estimate: {stats.get('calls_handled', 0)} AI-handled calls "
        "× human-agent cost − AI API costs."
    )

    if costs:
        st.bar_chart(
            pd.DataFrame(
                {
                    "cost": {
                        "Groq": costs.get("groq", 0),
                        "Deepgram": costs.get("deepgram", 0),
                        "ElevenLabs": costs.get("elevenlabs", 0),
                        "Twilio": costs.get("twilio", 0),
                    }
                }
            )
        )


if __name__ == "__main__":
    show_analytics_dashboard()
