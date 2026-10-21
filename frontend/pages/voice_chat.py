"""Streamlit Voice Chat tab: record → Deepgram → agent → ElevenLabs."""

from __future__ import annotations

import base64
import uuid
from typing import Any

import requests
import streamlit as st

API_BASE = "http://localhost:8000"
REQUEST_TIMEOUT = 180


def _ensure_state() -> None:
    if "voice_session_id" not in st.session_state:
        st.session_state["voice_session_id"] = str(uuid.uuid4())
    if "voice_messages" not in st.session_state:
        st.session_state["voice_messages"] = []
    if "voice_last_meta" not in st.session_state:
        st.session_state["voice_last_meta"] = None


def show_voice_chat() -> None:
    """Render the voice chat experience inside the main dashboard tab."""
    _ensure_state()

    st.markdown("## Voice Chat with Alex")
    st.caption("Speak your question or request")
    st.info("Audio is streamed directly to our partners, not stored locally.")

    # SECTION 1 — RECORD
    st.markdown("### Record")
    audio_value = st.audio_input("Tap to record a message for Alex")

    # SECTION 2 — SEND
    st.markdown("### Send")
    send = st.button("Send Voice Message", type="primary", use_container_width=True)

    if send:
        if not audio_value:
            st.warning("Record a message first.")
            return

        audio_bytes = audio_value.getvalue()
        with st.spinner("Transcribing..."):
            try:
                response = requests.post(
                    f"{API_BASE}/audio/chat",
                    files={"audio": ("voice.wav", audio_bytes, "audio/wav")},
                    data={"session_id": st.session_state["voice_session_id"]},
                    timeout=REQUEST_TIMEOUT,
                )
            except requests.exceptions.Timeout:
                st.error("Sorry, I didn't catch that. Try again.")
                return
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach the API at {API_BASE}. Is the FastAPI server running?")
                return
            except requests.exceptions.RequestException as exc:
                st.error(f"Network error: {exc}")
                return

        if response.status_code == 422:
            st.warning("Sorry, I didn't catch that. Try again.")
            return
        if response.status_code == 503:
            st.error(response.json().get("detail", "Voice services are not configured."))
            return
        if response.status_code >= 400:
            detail = ""
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            st.error(f"Voice chat failed: {detail}")
            return

        data: dict[str, Any] = response.json()
        st.session_state["voice_session_id"] = data.get(
            "session_id", st.session_state["voice_session_id"]
        )
        st.session_state["voice_messages"].append(
            {"role": "user", "content": data.get("transcript", "")}
        )
        st.session_state["voice_messages"].append(
            {
                "role": "assistant",
                "content": data.get("response", ""),
                "audio": data.get("audio", ""),
            }
        )
        st.session_state["voice_last_meta"] = {
            "intent": data.get("intent", "unknown"),
            "confidence": float(data.get("confidence", 0.0)),
            "action_taken": data.get("action_taken", "none"),
            "needs_escalation": data.get("needs_escalation", False),
        }

    # SECTION 3 — LISTEN / HISTORY
    st.markdown("### Listen")
    meta = st.session_state["voice_last_meta"]
    if meta:
        cols = st.columns(3)
        cols[0].metric("Intent", meta.get("intent", "-"))
        cols[1].metric("Confidence", f"{float(meta.get('confidence', 0.0)):.2f}")
        cols[2].metric("Action", meta.get("action_taken", "-"))

    if not st.session_state["voice_messages"]:
        st.caption("Your voice conversation will appear here.")
        return

    for message in st.session_state["voice_messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            audio_b64 = message.get("audio") or ""
            if audio_b64:
                try:
                    st.audio(base64.b64decode(audio_b64), format="audio/mp3")
                except Exception:
                    st.caption(
                        "I had trouble speaking. Please read my text response."
                    )
            elif message["role"] == "assistant":
                st.caption(
                    "I had trouble speaking. Please read my text response."
                )

    if st.button("Clear voice conversation"):
        st.session_state["voice_session_id"] = str(uuid.uuid4())
        st.session_state["voice_messages"] = []
        st.session_state["voice_last_meta"] = None
        st.rerun()


# Streamlit multipage entry when this file is opened from the sidebar.
if __name__ == "__main__":
    show_voice_chat()
