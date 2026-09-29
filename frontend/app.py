"""Streamlit dashboard for the voice agent.

Talks to the FastAPI backend over HTTP, so start the API first:
    uvicorn backend.main:app --reload
    streamlit run frontend/app.py
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API_URL = os.getenv("AGENT_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 90.0

st.set_page_config(page_title="Voice Agent Dashboard", page_icon="🎧", layout="wide")


def api_get(path: str) -> Any:
    """GET a backend endpoint, showing an error in the UI when it fails."""
    try:
        response = httpx.get(f"{API_URL}{path}", timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"Could not reach {path}: {exc}")
        return None


def api_post(path: str, payload: dict[str, Any] | None = None) -> Any:
    """POST to a backend endpoint, showing an error in the UI when it fails."""
    try:
        response = httpx.post(f"{API_URL}{path}", json=payload or {}, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"Request to {path} failed: {exc}")
        return None


def confidence_color(confidence: float) -> str:
    """Green above 0.7, amber from 0.4 to 0.7, red below 0.4."""
    if confidence > 0.7:
        return "#2ecc71"
    if confidence >= 0.4:
        return "#f1c40f"
    return "#e74c3c"


def init_state() -> None:
    """Create a session id and empty chat log on first load."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "last_meta" not in st.session_state:
        st.session_state.last_meta = {"intent": "-", "confidence": 0.0, "action_taken": "-"}


def render_sidebar() -> None:
    """Show the live session id, intent, confidence, and last action."""
    meta = st.session_state.last_meta
    confidence = float(meta.get("confidence", 0.0))

    with st.sidebar:
        st.subheader("Session")
        new_id = st.text_input("Session ID", value=st.session_state.session_id)
        if new_id != st.session_state.session_id:
            st.session_state.session_id = new_id
            st.session_state.chat = []
        if st.button("New session", width="stretch"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.chat = []
            st.session_state.last_meta = {"intent": "-", "confidence": 0.0, "action_taken": "-"}
            st.rerun()

        st.divider()
        st.subheader("Last turn")
        st.metric("Intent", meta.get("intent", "-"))
        st.markdown(
            f"<div style='font-size:0.85rem;color:#9aa4b2;'>Confidence</div>"
            f"<div style='font-size:1.6rem;font-weight:600;color:{confidence_color(confidence)};'>"
            f"{confidence:.2f}</div>",
            unsafe_allow_html=True,
        )
        st.progress(min(1.0, max(0.0, confidence)))
        st.caption(f"Action: {meta.get('action_taken', '-')}")
        if meta.get("needs_escalation"):
            st.warning("This turn was handed to a human.")

        st.divider()
        health = api_get("/health")
        if health:
            st.caption(f"Model: {health.get('model')}")
            if not health.get("groq_api_key_configured"):
                st.error("GROQ_API_KEY is not set in .env")
        if st.button("Re-ingest knowledge base", width="stretch"):
            with st.spinner("Re-indexing documents..."):
                result = api_post("/ingest")
            if result:
                st.success(f"Indexed {result['chunks']} chunks from {result['documents']} documents.")


def render_chat_tab() -> None:
    """Chat transcript plus the input box."""
    st.subheader("Chat with Alex")
    for turn in st.session_state.chat:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])

    message = st.chat_input("Ask about hours, services, or book an appointment")
    if not message:
        return

    st.session_state.chat.append({"role": "user", "content": message})
    with st.chat_message("user"):
        st.write(message)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            data = api_post(
                "/chat",
                {"session_id": st.session_state.session_id, "message": message},
            )
        if not data:
            return
        st.write(data["response"])

    st.session_state.chat.append({"role": "assistant", "content": data["response"]})
    st.session_state.last_meta = {
        "intent": data["intent"],
        "confidence": data["confidence"],
        "action_taken": data["action_taken"],
        "needs_escalation": data.get("needs_escalation", False),
    }
    st.rerun()


def render_analytics_tab() -> None:
    """Totals, intent breakdown, bookings, and the escalation log."""
    st.subheader("Analytics")
    if st.button("Refresh"):
        st.rerun()

    data = api_get("/analytics")
    if not data:
        return

    columns = st.columns(5)
    columns[0].metric("Conversations today", data["conversations_today"])
    columns[1].metric("Total conversations", data["total_conversations"])
    columns[2].metric("Appointments", data["total_appointments"])
    columns[3].metric("Escalations", data["total_escalations"])
    columns[4].metric("Avg confidence", f"{data['average_confidence']:.2f}")

    st.markdown("#### Intents")
    breakdown = data["intent_breakdown"]
    if breakdown:
        st.bar_chart(pd.DataFrame({"count": breakdown}))
    else:
        st.caption("No conversations yet.")

    st.markdown("#### Recent appointments")
    appointments = data["recent_appointments"]
    if appointments:
        st.dataframe(pd.DataFrame(appointments), width="stretch", hide_index=True)
    else:
        st.caption("No appointments booked yet.")

    st.markdown("#### Escalation log")
    escalations = data["recent_escalations"]
    if escalations:
        st.dataframe(pd.DataFrame(escalations), width="stretch", hide_index=True)
    else:
        st.caption("No escalations logged yet.")


def main() -> None:
    """Render the dashboard."""
    init_state()
    st.title("AI Voice Agent — Reception Dashboard")
    render_sidebar()
    chat_tab, analytics_tab = st.tabs(["Chat", "Analytics"])
    with chat_tab:
        render_chat_tab()
    with analytics_tab:
        render_analytics_tab()


main()
