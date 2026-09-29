"""LangChain tools the agent can call.

Appointments and escalations are written to JSON files next to the
project root, which stand in for a real CRM and ticketing system.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import date as date_cls
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from backend.config import settings
from backend.retrieval.vectorstore import hybrid_search

logger = logging.getLogger(__name__)

# Guards read-modify-write of the JSON files when requests overlap.
_file_lock = threading.Lock()

WEEKDAY_SLOTS = [
    "8:00 AM",
    "8:30 AM",
    "9:00 AM",
    "9:30 AM",
    "10:00 AM",
    "10:30 AM",
    "1:00 PM",
    "1:30 PM",
    "2:00 PM",
    "2:30 PM",
    "3:00 PM",
    "3:30 PM",
    "4:00 PM",
]
SATURDAY_SLOTS = ["9:00 AM", "10:00 AM", "11:00 AM", "12:00 PM"]


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    """Read a JSON array from disk, returning [] when it is missing or corrupt."""
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
    """Append one record to a JSON array file, creating the file if needed."""
    with _file_lock:
        records = _read_json_list(path)
        records.append(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)


def load_appointments() -> list[dict[str, Any]]:
    """Return every booked appointment, oldest first."""
    return _read_json_list(settings.appointments_path)


def load_escalations() -> list[dict[str, Any]]:
    """Return every logged escalation, oldest first."""
    return _read_json_list(settings.escalations_path)


def parse_date(value: str) -> date_cls | None:
    """Parse a date string in one of the formats callers commonly give.

    Returns None when the text is not a date the clinic can book against.
    """
    cleaned = value.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%B %d, %Y", "%B %d %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


@tool
def search_knowledge_base(query: str) -> str:
    """Search the clinic knowledge base and return the most relevant passages.

    Use this before answering any question about hours, services, policies,
    or how booking works. Returns numbered passages with their source file.
    """
    try:
        result = hybrid_search(query, top_k=settings.retrieval_top_k)
    except Exception as exc:
        logger.error("Knowledge base search failed for %r: %s", query, exc)
        return f"Knowledge base search failed: {exc}"

    if not result.chunks:
        return "No relevant information found in the knowledge base."

    sections = [
        f"[{position}] (source: {chunk.source}, similarity: {chunk.cosine_similarity:.2f})\n"
        f"{chunk.text}"
        for position, chunk in enumerate(result.chunks, start=1)
    ]
    return "\n\n".join(sections)


@tool
def get_available_slots(date: str) -> str:
    """List the appointment times open on a given date.

    Accepts dates like 2026-10-05 or 10/05/2026. Sundays are closed and
    Saturdays only offer urgent same-week visits.
    """
    parsed = parse_date(date)
    if parsed is None:
        return (
            f"'{date}' is not a date I can read. "
            "Please give a date like 2026-10-05."
        )

    weekday = parsed.weekday()
    if weekday == 6:
        return f"The clinic is closed on Sunday {parsed.isoformat()}."
    slots = SATURDAY_SLOTS if weekday == 5 else WEEKDAY_SLOTS

    # Mock availability: times already booked on that date are removed.
    taken = {
        appointment.get("time")
        for appointment in load_appointments()
        if appointment.get("date") == parsed.isoformat()
    }
    open_slots = [slot for slot in slots if slot not in taken]
    if not open_slots:
        return f"No times are open on {parsed.isoformat()}. Please try another date."
    return f"Open times on {parsed.isoformat()}: {', '.join(open_slots)}."


@tool
def book_appointment(name: str, date: str, time: str, reason: str) -> str:
    """Book an appointment and return a confirmation with a booking ID.

    All four details are required. The booking is saved to appointments.json,
    which stands in for the clinic CRM.
    """
    missing = [
        field
        for field, value in (("name", name), ("date", date), ("time", time), ("reason", reason))
        if not str(value).strip()
    ]
    if missing:
        return f"Cannot book yet. Missing: {', '.join(missing)}."

    parsed = parse_date(date)
    if parsed is None:
        return f"'{date}' is not a date I can read. Please give a date like 2026-10-05."
    if parsed.weekday() == 6:
        return f"The clinic is closed on Sunday {parsed.isoformat()}. Please pick another day."

    booking_id = f"APT-{uuid.uuid4().hex[:8].upper()}"
    record = {
        "booking_id": booking_id,
        "name": name.strip(),
        "date": parsed.isoformat(),
        "time": time.strip(),
        "reason": reason.strip(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        _append_json_record(settings.appointments_path, record)
    except OSError as exc:
        logger.error("Failed to save appointment: %s", exc)
        return f"Something went wrong saving the appointment: {exc}"

    logger.info("Booked %s for %s on %s at %s", booking_id, record["name"], record["date"], record["time"])
    return (
        f"Booked. {record['name']} on {record['date']} at {record['time']} "
        f"for {record['reason']}. Your booking ID is {booking_id}."
    )


@tool
def escalate_to_human(reason: str, summary: str) -> str:
    """Hand the conversation to a human and log why.

    Use when the knowledge base cannot answer confidently, the caller asks
    for a person, or the request is outside what the receptionist handles.
    """
    escalation_id = f"ESC-{uuid.uuid4().hex[:8].upper()}"
    record = {
        "escalation_id": escalation_id,
        "reason": reason.strip(),
        "summary": summary.strip(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        _append_json_record(settings.escalations_path, record)
    except OSError as exc:
        logger.error("Failed to log escalation: %s", exc)

    logger.info("Escalation %s logged: %s", escalation_id, record["reason"])
    return (
        "I want to make sure you get an accurate answer, so I am passing this to "
        f"one of our staff members. They will follow up shortly. Your reference is {escalation_id}."
    )


ALL_TOOLS = [
    search_knowledge_base,
    get_available_slots,
    book_appointment,
    escalate_to_human,
]
