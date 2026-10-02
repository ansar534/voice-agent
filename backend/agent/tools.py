"""Actions the agent can take: search, book, check slots, escalate.

These are plain functions returning plain strings, so they can be called
directly from graph nodes without any tool-calling round trip.
Appointments and escalations are stored as JSON files standing in for a
real CRM and ticketing system.
"""

from __future__ import annotations

import json
import logging
import random
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import APPOINTMENTS_PATH, ESCALATIONS_PATH
from ..retrieval.vectorstore import get_retriever

logger = logging.getLogger(__name__)

# Serializes the read-modify-write cycle on the JSON files.
_file_lock = threading.Lock()

MOCK_SLOTS = ["9:00 AM", "10:00 AM", "11:00 AM", "2:00 PM", "3:00 PM", "4:00 PM"]


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    """Read a JSON array, returning [] when the file is missing or invalid."""
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
    """Append one record to a JSON array file, creating it when needed."""
    with _file_lock:
        records = _read_json_list(path)
        records.append(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)


def search_knowledge_base(query: str) -> dict[str, Any]:
    """Search the knowledge base and return formatted context plus confidence.

    Returns {"context": str, "confidence": float, "sources": list[str]}.
    On failure the context explains the problem and confidence is 0.0,
    which pushes the agent to escalate rather than invent an answer.
    """
    try:
        retriever = get_retriever()
        result = retriever.retrieve(query, n_results=5)
        context = retriever.format_context(result["chunks"], result["sources"])
        return {
            "context": context,
            "confidence": float(result["confidence"]),
            "sources": result["sources"],
        }
    except Exception as exc:
        logger.error("Knowledge base search failed for %r: %s", query, exc)
        return {
            "context": "The knowledge base is unavailable right now.",
            "confidence": 0.0,
            "sources": [],
        }


def book_appointment(name: str, date: str, time: str, reason: str) -> str:
    """Save an appointment and return a confirmation with its booking ID.

    The ID has the form CMC-123456. The record is appended to
    appointments.json, which acts as the clinic CRM for Phase 1.
    """
    missing = [
        field
        for field, value in (("name", name), ("date", date), ("time", time), ("reason", reason))
        if not str(value).strip()
    ]
    if missing:
        return f"I still need your {', '.join(missing)} before I can book that."

    booking_id = f"CMC-{random.randint(100000, 999999)}"
    record = {
        "booking_id": booking_id,
        "name": name.strip(),
        "date": date.strip(),
        "time": time.strip(),
        "reason": reason.strip(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    try:
        _append_json_record(APPOINTMENTS_PATH, record)
    except OSError as exc:
        logger.error("Failed to save appointment: %s", exc)
        return "I could not save that appointment. Let me get a staff member to help you."

    logger.info("Booked %s for %s on %s at %s", booking_id, record["name"], date, time)
    return (
        f"You're all set, {record['name']}. I have you down for {record['date']} "
        f"at {record['time']} for {record['reason']}. "
        f"Your booking ID is {booking_id}. Please arrive 10 minutes early."
    )


def get_available_slots(date: str) -> str:
    """List the times still open on a date, as a readable sentence.

    Any slot already present in appointments.json for that date is
    removed from the list.
    """
    if not date.strip():
        return "What date were you thinking of?"

    taken = {
        str(appointment.get("time", "")).strip()
        for appointment in _read_json_list(APPOINTMENTS_PATH)
        if str(appointment.get("date", "")).strip() == date.strip()
    }
    open_slots = [slot for slot in MOCK_SLOTS if slot not in taken]

    if not open_slots:
        return f"I'm fully booked on {date}. Would another day work for you?"
    if len(open_slots) == 1:
        return f"On {date} I only have {open_slots[0]} left."
    listed = f"{', '.join(open_slots[:-1])}, and {open_slots[-1]}"
    return f"On {date} I have {listed} open."


def escalate_to_human(reason: str, conversation_summary: str) -> str:
    """Log a handoff to staff and return the message for the caller."""
    record = {
        "escalation_id": f"ESC-{random.randint(100000, 999999)}",
        "reason": reason.strip(),
        "summary": conversation_summary.strip(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    try:
        _append_json_record(ESCALATIONS_PATH, record)
    except OSError as exc:
        logger.error("Failed to log escalation: %s", exc)

    logger.info("Escalation %s logged: %s", record["escalation_id"], record["reason"])
    return (
        "I want to make sure you get an accurate answer, so I'm connecting you "
        "with one of our staff members. They'll follow up with you shortly. "
        f"Your reference number is {record['escalation_id']}."
    )


def get_appointments() -> list[dict[str, Any]]:
    """Return every appointment currently stored, oldest first."""
    return _read_json_list(APPOINTMENTS_PATH)


def get_escalations() -> list[dict[str, Any]]:
    """Return every logged escalation, oldest first.

    Records written by earlier versions stored the time under
    "created_at", so that key is accepted as a fallback and the file
    does not have to be deleted after an upgrade.
    """
    records = _read_json_list(ESCALATIONS_PATH)
    for record in records:
        if "timestamp" not in record:
            record["timestamp"] = record.get("created_at", "")
    return records
