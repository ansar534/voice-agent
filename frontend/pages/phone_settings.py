"""Streamlit Phone Settings tab for Twilio configuration and call history."""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests
import streamlit as st

API_BASE = "http://localhost:8000"
REQUEST_TIMEOUT = 60

GREETING_OPTIONS = [
    "Thank you for calling City Medical Clinic. This is Alex, the AI receptionist. How can I help you today?",
    "Hello, you've reached City Medical Clinic. I'm Alex. I can answer questions or book an appointment.",
    "Hi, this is Alex at City Medical Clinic. Tell me how I can help.",
]

ESCALATION_OPTIONS = {
    "offer_callback": "Offer a staff callback and end the call",
    "transfer_message": "Say a hold / transfer message and end the call",
}


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


def _put(path: str, payload: dict) -> Any | None:
    try:
        response = requests.put(
            f"{API_BASE}{path}", json=payload, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        st.error(f"Failed to save settings: {exc}")
    return None


def _post(path: str, payload: dict) -> Any | None:
    try:
        response = requests.post(
            f"{API_BASE}{path}", json=payload, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        st.error(f"Request failed: {exc}")
    return None


def show_phone_settings() -> None:
    """Render Twilio number info, settings, test call, and call history."""
    st.markdown("## Phone Settings")
    st.caption("Route clinic calls through Alex via Twilio")

    info = _get("/phone/info") or {}
    phone_number = info.get("phone_number") or "(not configured)"
    configured = bool(info.get("configured"))

    st.markdown(f"**Your Twilio phone number:** `{phone_number}`")
    if configured:
        st.success("Twilio credentials detected.")
    else:
        st.warning(
            "Add TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER to `.env`."
        )

    st.info(
        f"Forward your clinic number to **{phone_number}**. "
        f"In the Twilio console, set the voice webhook to "
        f"`{info.get('incoming_webhook', 'http://localhost:8000/phone/incoming')}`."
    )

    settings = info.get("settings") or _get("/phone/settings") or {}
    st.markdown("### Settings")
    accept = st.toggle(
        "Accept incoming calls",
        value=bool(settings.get("accept_incoming_calls", True)),
    )
    record = st.toggle(
        "Record all calls",
        value=bool(settings.get("record_all_calls", True)),
    )

    current_greeting = settings.get("default_greeting") or GREETING_OPTIONS[0]
    greeting_choices = list(dict.fromkeys([current_greeting, *GREETING_OPTIONS]))
    greeting = st.selectbox(
        "Default greeting message",
        options=greeting_choices,
        index=0,
    )
    escalation = st.selectbox(
        "Escalation behavior",
        options=list(ESCALATION_OPTIONS.keys()),
        format_func=lambda key: ESCALATION_OPTIONS[key],
        index=0
        if settings.get("escalation_behavior") not in ESCALATION_OPTIONS
        else list(ESCALATION_OPTIONS.keys()).index(settings.get("escalation_behavior")),
    )

    if st.button("Save phone settings", type="primary"):
        saved = _put(
            "/phone/settings",
            {
                "accept_incoming_calls": accept,
                "record_all_calls": record,
                "default_greeting": greeting,
                "escalation_behavior": escalation,
            },
        )
        if saved:
            st.success("Phone settings saved.")

    st.markdown("### Test call")
    to_number = st.text_input("Destination number (E.164, e.g. +15551234567)")
    if st.button("Make a test call"):
        if not to_number.strip():
            st.warning("Enter a phone number first.")
        else:
            result = _post("/phone/test-call", {"to_number": to_number.strip()})
            if result:
                st.success(f"Test call started: {result.get('call_sid')}")

    st.markdown("### Call quality")
    analytics = _get("/analytics")
    if analytics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total calls", analytics.get("total_calls", 0))
        c2.metric("Missed calls", analytics.get("missed_calls", 0))
        c3.metric("Avg duration (s)", analytics.get("avg_call_duration", 0))
        quality = analytics.get("call_quality") or {}
        c4.metric("Resolution rate", f"{float(quality.get('resolution_rate', 0)) * 100:.0f}%")

    st.markdown("### Call history")
    calls = _get("/calls")
    if calls:
        st.dataframe(pd.DataFrame(calls), use_container_width=True, hide_index=True)
    elif calls is not None:
        st.caption("No phone calls logged yet.")


if __name__ == "__main__":
    show_phone_settings()
