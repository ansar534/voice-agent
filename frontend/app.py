"""Streamlit dashboard for the City Medical Clinic voice agent.

Start the API first, then run:
    python -m streamlit run frontend/app.py
"""

from __future__ import annotations

import uuid
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_BASE = "http://localhost:8000"
REQUEST_TIMEOUT = 120

INTENT_COLORS = {
    "faq": "#3b82f6",
    "book_appointment": "#22c55e",
    "escalate": "#ef4444",
    "unknown": "#6b7280",
}

st.set_page_config(page_title="Alex | Clinic Receptionist", page_icon="🩺", layout="wide")


def api_get(path: str) -> Any | None:
    """GET a backend endpoint, surfacing any failure in the UI."""
    try:
        response = requests.get(f"{API_BASE}{path}", timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach the API at {API_BASE}. Is the FastAPI server running?")
    except requests.exceptions.Timeout:
        st.error(f"Request to {path} timed out.")
    except requests.exceptions.RequestException as exc:
        st.error(f"Request to {path} failed: {exc}")
    return None


def api_post(path: str, payload: dict | None = None) -> Any | None:
    """POST to a backend endpoint, surfacing any failure in the UI."""
    try:
        response = requests.post(
            f"{API_BASE}{path}", json=payload or {}, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach the API at {API_BASE}. Is the FastAPI server running?")
    except requests.exceptions.Timeout:
        st.error(f"Request to {path} timed out.")
    except requests.exceptions.RequestException as exc:
        st.error(f"Request to {path} failed: {exc}")
    return None


def init_session_state() -> None:
    """Create the session id, chat log, and metadata on first load."""
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "last_meta" not in st.session_state:
        st.session_state["last_meta"] = None


def confidence_color(confidence: float) -> str:
    """Green above 0.7, yellow from 0.4 to 0.7, red below 0.4."""
    if confidence > 0.7:
        return "#22c55e"
    if confidence >= 0.4:
        return "#eab308"
    return "#ef4444"


def render_sidebar() -> None:
    """Show intent, confidence, action, and session id for the last turn."""
    with st.sidebar:
        st.markdown("### Last message")
        meta = st.session_state["last_meta"]

        if not meta:
            st.caption("Send a message to see how Alex handled it.")
        else:
            intent = meta.get("intent", "unknown")
            confidence = float(meta.get("confidence", 0.0))

            st.markdown(
                f"<span style='background-color:{INTENT_COLORS.get(intent, '#6b7280')};"
                "color:white;padding:4px 12px;border-radius:12px;font-size:0.85rem;"
                f"font-weight:600;'>{intent}</span>",
                unsafe_allow_html=True,
            )

            st.markdown(
                "<div style='margin-top:16px;font-size:0.8rem;color:#9ca3af;'>Confidence</div>"
                f"<div style='font-size:1.8rem;font-weight:700;"
                f"color:{confidence_color(confidence)};'>{confidence:.2f}</div>",
                unsafe_allow_html=True,
            )
            st.progress(min(1.0, max(0.0, confidence)))

            st.markdown(
                "<div style='margin-top:12px;font-size:0.8rem;color:#9ca3af;'>Action taken</div>"
                f"<div style='font-size:1rem;'>{meta.get('action_taken', '-')}</div>",
                unsafe_allow_html=True,
            )

            if meta.get("needs_escalation"):
                st.warning("Handed off to a human")

        st.divider()
        if st.button("Start new conversation", use_container_width=True):
            st.session_state["session_id"] = str(uuid.uuid4())
            st.session_state["messages"] = []
            st.session_state["last_meta"] = None
            st.rerun()

        st.markdown(
            "<div style='font-size:0.7rem;color:#6b7280;margin-top:8px;'>Session "
            f"{st.session_state['session_id']}</div>",
            unsafe_allow_html=True,
        )


def render_chat_tab() -> None:
    """Render the transcript and handle new messages."""
    st.markdown("## Chat with Alex")
    st.caption("City Medical Clinic AI Receptionist")

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Ask about hours, services, or book an appointment")
    if not prompt:
        return

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Alex is typing..."):
            data = api_post(
                "/chat",
                {"session_id": st.session_state["session_id"], "message": prompt},
            )
        if data is None:
            # Drop the unanswered message so a retry does not duplicate it.
            st.session_state["messages"].pop()
            return
        st.write(data["response"])

    st.session_state["messages"].append({"role": "assistant", "content": data["response"]})
    st.session_state["session_id"] = data["session_id"]
    st.session_state["last_meta"] = {
        "intent": data["intent"],
        "confidence": data["confidence"],
        "action_taken": data["action_taken"],
        "needs_escalation": data.get("needs_escalation", False),
    }
    st.rerun()


def render_analytics_tab() -> None:
    """Show aggregate stats, the intent chart, and booked appointments."""
    st.markdown("## Analytics")

    header, refresh = st.columns([5, 1])
    with refresh:
        if st.button("Refresh", use_container_width=True):
            st.rerun()

    stats = api_get("/analytics")
    if stats is None:
        return

    one, two, three, four = st.columns(4)
    one.metric("Total Sessions", stats["total_sessions"])
    two.metric("Top Intent", stats["top_intent"])
    three.metric("Avg Turns", stats["avg_turns"])
    four.metric("Escalation Rate", f"{stats['escalation_rate'] * 100:.0f}%")

    five, six, seven = st.columns(3)
    five.metric("Appointments", stats["total_appointments"])
    six.metric("Escalations", stats["total_escalations"])
    seven.metric("Avg Confidence", f"{stats['avg_confidence']:.2f}")

    st.markdown("### Intents")
    intent_counts = stats.get("intent_counts") or {}
    if intent_counts:
        st.bar_chart(pd.DataFrame({"count": intent_counts}))
    else:
        st.caption("No conversations yet.")

    st.markdown("### Appointments")
    appointments = api_get("/appointments")
    if appointments:
        st.dataframe(pd.DataFrame(appointments), use_container_width=True, hide_index=True)
    elif appointments is not None:
        st.caption("No appointments booked yet.")

    st.markdown("### Escalations")
    escalations = api_get("/escalations")
    if escalations:
        st.dataframe(pd.DataFrame(escalations), use_container_width=True, hide_index=True)
    elif escalations is not None:
        st.caption("No escalations logged yet.")

    st.divider()
    if st.button("Re-ingest Knowledge Base"):
        with st.spinner("Rebuilding the knowledge base indexes..."):
            result = api_post("/ingest")
        if result:
            st.success(
                f"Indexed {result['chunks_stored']} chunks from "
                f"{result['documents']} documents."
            )


def main() -> None:
    """Draw the dashboard."""
    init_session_state()
    st.title("AI Voice Agent Dashboard")

    render_sidebar()
    chat_tab, analytics_tab = st.tabs(["Chat", "Analytics"])
    with chat_tab:
        render_chat_tab()
    with analytics_tab:
        render_analytics_tab()


main()
