"""Post-call automation: summaries, email, SMS, tickets, and metrics."""

from __future__ import annotations

import json
import logging
import random
import smtplib
import threading
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from ..config import (
    BUSINESS_NAME,
    CALL_ANALYTICS_PATH,
    CLINIC_ADDRESS,
    CLINIC_STAFF_EMAIL,
    SMTP_FROM,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_USER,
    SUPPORT_TICKETS_PATH,
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_PHONE_NUMBER,
    twilio_configured,
)

logger = logging.getLogger(__name__)

_file_lock = threading.Lock()


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Could not read %s: %s", path, exc)
        return []
    return data if isinstance(data, list) else []


def _append_json_record(path: Path, record: dict[str, Any]) -> None:
    with _file_lock:
        records = _read_json_list(path)
        records.append(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)


def get_support_tickets() -> list[dict[str, Any]]:
    """Return every support ticket on disk."""
    return _read_json_list(SUPPORT_TICKETS_PATH)


def get_call_analytics_records() -> list[dict[str, Any]]:
    """Return every per-call analytics row on disk."""
    return _read_json_list(CALL_ANALYTICS_PATH)


class PostCallAutomation:
    """Runs after every phone call ends."""

    def on_call_end(self, session_id: str, call_data: dict[str, Any]) -> dict[str, Any]:
        """Summarize the call, notify staff when needed, and log metrics."""
        messages = call_data.get("messages") or []
        transcript = "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in messages
        )
        booking_info = call_data.get("booking_info") or {}
        needs_escalation = bool(call_data.get("needs_escalation"))
        action_taken = str(call_data.get("action_taken", "none"))
        intent = str(call_data.get("intent", "unknown"))
        from_number = str(call_data.get("from_number", ""))

        caller_name = str(booking_info.get("name") or "unknown caller")
        reason = str(
            booking_info.get("reason")
            or intent
            or "general inquiry"
        )
        resolution = (
            "escalated to staff"
            if needs_escalation
            else ("appointment booked" if action_taken == "appointment_booked" else "resolved by AI")
        )

        summary = (
            f"Call with {caller_name} about {reason}. "
            f"The agent classified the intent as {intent} and the outcome was {resolution}. "
            f"Session {session_id} ended with action '{action_taken}'."
        )

        results: dict[str, Any] = {
            "summary": summary,
            "email_sent": False,
            "sms_sent": False,
            "ticket_id": None,
        }

        if needs_escalation:
            results["email_sent"] = self.send_support_email(summary, transcript)
            ticket = self.create_support_ticket(
                {
                    "reason": reason,
                    "transcript": transcript,
                    "summary": summary,
                    "session_id": session_id,
                    "from_number": from_number,
                }
            )
            results["ticket_id"] = ticket.get("ticket_id")

        if action_taken == "appointment_booked" and from_number:
            results["sms_sent"] = self.send_confirmation_sms(from_number, booking_info)

        follow_up_needed = needs_escalation or action_taken in {
            "llm_unavailable",
            "agent_error",
            "clarified",
        }
        if follow_up_needed and not results["ticket_id"]:
            ticket = self.create_support_ticket(
                {
                    "reason": f"Follow-up needed ({action_taken})",
                    "transcript": transcript,
                    "summary": summary,
                    "session_id": session_id,
                    "from_number": from_number,
                }
            )
            results["ticket_id"] = ticket.get("ticket_id")

        metrics = {
            "session_id": session_id,
            "call_sid": call_data.get("call_sid", ""),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "call_duration": int(call_data.get("duration", 0) or 0),
            "intents_covered": [intent],
            "booking_success": action_taken == "appointment_booked",
            "escalation_reason": reason if needs_escalation else "",
            "resolved": bool(call_data.get("resolved", not needs_escalation)),
            "confidence": float(call_data.get("confidence", 0.0) or 0.0),
            "summary": summary,
        }
        try:
            _append_json_record(CALL_ANALYTICS_PATH, metrics)
        except OSError as exc:
            logger.error("Failed to save call analytics: %s", exc)

        logger.info(
            "Post-call complete for %s: escalation=%s booking=%s ticket=%s",
            session_id,
            needs_escalation,
            action_taken == "appointment_booked",
            results["ticket_id"],
        )
        return results

    def send_support_email(self, summary: str, transcript: str) -> bool:
        """Email clinic staff about an escalation via SMTP.

        When SMTP is not configured the email is logged instead of sent so
        local development still exercises the escalation path.
        """
        timestamp = datetime.now().isoformat(timespec="seconds")
        subject = f"AI Agent Escalation: {timestamp}"
        body = (
            f"Business: {BUSINESS_NAME}\n\n"
            f"Summary:\n{summary}\n\n"
            f"Full transcript:\n{transcript}\n"
        )
        recipient = CLINIC_STAFF_EMAIL or SMTP_USER

        if not SMTP_HOST or not recipient:
            logger.warning(
                "SMTP not configured; escalation email logged only.\nSubject: %s\n%s",
                subject,
                body[:500],
            )
            return False

        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = SMTP_FROM
        message["To"] = recipient
        message.set_content(body)

        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
                server.starttls()
                if SMTP_USER and SMTP_PASSWORD:
                    server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(message)
            logger.info("Escalation email sent to %s", recipient)
            return True
        except Exception as exc:
            logger.error("Failed to send escalation email: %s", exc)
            return False

    def send_confirmation_sms(self, phone_number: str, appointment_info: dict) -> bool:
        """Send an appointment confirmation SMS through Twilio."""
        date = appointment_info.get("date", "your scheduled date")
        time_slot = appointment_info.get("time", "your scheduled time")
        message = (
            f"Your appointment is confirmed: {date} at {time_slot}. "
            f"Address: {CLINIC_ADDRESS}. — {BUSINESS_NAME}"
        )

        if not twilio_configured():
            logger.warning("Twilio not configured; SMS logged only: %s -> %s", phone_number, message)
            return False

        try:
            from twilio.rest import Client

            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            client.messages.create(
                body=message,
                from_=TWILIO_PHONE_NUMBER,
                to=phone_number,
            )
            logger.info("Confirmation SMS sent to %s", phone_number)
            return True
        except Exception as exc:
            logger.error("Failed to send confirmation SMS: %s", exc)
            return False

    def create_support_ticket(self, data: dict) -> dict[str, Any]:
        """Save a support ticket to support_tickets.json."""
        ticket = {
            "ticket_id": f"TKT-{random.randint(100000, 999999)}",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "reason": str(data.get("reason", "unspecified")).strip(),
            "transcript": str(data.get("transcript", "")).strip(),
            "summary": str(data.get("summary", "")).strip(),
            "session_id": str(data.get("session_id", "")),
            "from_number": str(data.get("from_number", "")),
            "call_sid": str(data.get("call_sid", "")),
            "status": "open",
        }
        try:
            _append_json_record(SUPPORT_TICKETS_PATH, ticket)
        except OSError as exc:
            logger.error("Failed to save support ticket: %s", exc)
        logger.info("Created support ticket %s", ticket["ticket_id"])
        return ticket
