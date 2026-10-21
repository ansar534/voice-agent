"""Twilio voice webhooks that drive Alex over the phone."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from twilio.rest import Client
from twilio.twiml.voice_response import Gather, VoiceResponse

from ..agent.graph import run_agent
from ..audio.speaker import get_speaker
from ..audio.transcriber import get_transcriber
from ..automation.post_call import PostCallAutomation
from ..config import (
    AUDIO_CACHE_DIR,
    BUSINESS_NAME,
    CALLS_PATH,
    PHONE_SETTINGS_PATH,
    PUBLIC_BASE_URL,
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_PHONE_NUMBER,
    elevenlabs_key_is_set,
    twilio_configured,
)
from ..memory.session import session_memory

logger = logging.getLogger(__name__)

_file_lock = threading.Lock()

DEFAULT_GREETING = (
    f"Thank you for calling {BUSINESS_NAME}. This is Alex, the AI receptionist. "
    "How can I help you today?"
)

DEFAULT_SETTINGS = {
    "accept_incoming_calls": True,
    "record_all_calls": True,
    "default_greeting": DEFAULT_GREETING,
    "escalation_behavior": "offer_callback",
}


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with _file_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def load_phone_settings() -> dict[str, Any]:
    """Return persisted phone settings merged with defaults."""
    if not PHONE_SETTINGS_PATH.is_file():
        return dict(DEFAULT_SETTINGS)
    try:
        with PHONE_SETTINGS_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULT_SETTINGS)
    if not isinstance(data, dict):
        return dict(DEFAULT_SETTINGS)
    merged = dict(DEFAULT_SETTINGS)
    merged.update(data)
    return merged


def save_phone_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Persist phone settings and return the merged result."""
    merged = dict(DEFAULT_SETTINGS)
    merged.update(settings or {})
    _write_json(PHONE_SETTINGS_PATH, merged)
    return merged


def get_calls() -> list[dict[str, Any]]:
    """Return every logged phone call, oldest first."""
    return _read_json_list(CALLS_PATH)


class TwilioPhoneAgent:
    """Handles Twilio voice webhooks and keeps call metadata for analytics."""

    def __init__(self) -> None:
        self.phone_number = TWILIO_PHONE_NUMBER
        self.client: Client | None = None
        if twilio_configured():
            self.client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        self.post_call = PostCallAutomation()

    def _session_id_for_caller(self, from_number: str) -> str:
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        safe = "".join(ch for ch in from_number if ch.isalnum()) or "unknown"
        return f"phone-{safe}-{stamp}"

    async def handle_incoming_call(self, from_number: str, to_number: str) -> str:
        """Build the initial TwiML for an inbound call.

        Creates a session keyed by caller + timestamp, greets the caller,
        and opens a Gather so Twilio streams speech back to /phone/gather.
        """
        settings = load_phone_settings()
        response = VoiceResponse()

        if not settings.get("accept_incoming_calls", True):
            response.say(
                f"Thank you for calling {BUSINESS_NAME}. "
                "We are not accepting AI-handled calls right now. "
                "Please try again during staffed hours."
            )
            response.hangup()
            return str(response)

        session_id = self._session_id_for_caller(from_number)
        session_memory.get_session(session_id)

        greeting = str(settings.get("default_greeting") or DEFAULT_GREETING)
        gather = Gather(
            input="speech",
            action=f"/phone/gather?session_id={quote(session_id)}",
            method="POST",
            timeout=5,
            speech_timeout="auto",
            language="en-US",
        )
        gather.say(greeting, voice="Polly.Joanna")
        response.append(gather)
        response.redirect(
            f"/phone/gather?session_id={quote(session_id)}&empty=1",
            method="POST",
        )

        self.save_call_recording(
            call_sid=f"pending-{session_id}",
            recording_url="",
            from_number=from_number,
            to_number=to_number,
            session_id=session_id,
            status="ringing",
            duration=0,
        )
        logger.info("Incoming call from %s -> session %s", from_number, session_id)
        return str(response)

    async def handle_gather(
        self,
        session_id: str,
        speech_result: str,
        call_sid: str = "",
        from_number: str = "",
        confidence: float | None = None,
    ) -> str:
        """Process one Gather result: agent reply + next Gather loop."""
        response = VoiceResponse()
        transcript = (speech_result or "").strip()

        if not transcript:
            gather = Gather(
                input="speech",
                action=f"/phone/gather?session_id={quote(session_id)}",
                method="POST",
                timeout=5,
                speech_timeout="auto",
                language="en-US",
            )
            gather.say(
                "Sorry, I didn't catch that. Could you say that again?",
                voice="Polly.Joanna",
            )
            response.append(gather)
            response.redirect(
                f"/phone/gather?session_id={quote(session_id)}&empty=1",
                method="POST",
            )
            return str(response)

        agent_result = run_agent(session_id, transcript)
        reply = str(agent_result.get("response", "")).strip() or (
            "I'm not sure I caught that. Could you tell me a bit more?"
        )

        # Prefer ElevenLabs audio when configured; fall back to Twilio <Say>.
        try:
            play_url = await self._store_speech_and_get_url(reply)
            response.play(play_url)
        except Exception as exc:
            logger.warning("ElevenLabs play failed, using Twilio Say: %s", exc)
            response.say(reply, voice="Polly.Joanna")

        if agent_result.get("needs_escalation"):
            settings = load_phone_settings()
            behavior = settings.get("escalation_behavior", "offer_callback")
            if behavior == "transfer_message":
                response.say(
                    "A staff member will take it from here. Please hold.",
                    voice="Polly.Joanna",
                )
            else:
                response.say(
                    "I've noted your request and a staff member will call you back shortly. "
                    "Goodbye.",
                    voice="Polly.Joanna",
                )
            response.hangup()
            self._finalize_call(
                session_id=session_id,
                call_sid=call_sid,
                from_number=from_number,
                agent_result=agent_result,
                resolved=False,
            )
            return str(response)

        gather = Gather(
            input="speech",
            action=f"/phone/gather?session_id={quote(session_id)}",
            method="POST",
            timeout=5,
            speech_timeout="auto",
            language="en-US",
        )
        gather.say("Is there anything else I can help with?", voice="Polly.Joanna")
        response.append(gather)
        response.say(
            "Thank you for calling. Goodbye.",
            voice="Polly.Joanna",
        )
        response.hangup()
        self._finalize_call(
            session_id=session_id,
            call_sid=call_sid,
            from_number=from_number,
            agent_result=agent_result,
            resolved=agent_result.get("action_taken") != "escalated",
        )
        return str(response)

    async def generate_twiml_response(self, text: str) -> str:
        """Convert agent text into TwiML that plays ElevenLabs audio."""
        response = VoiceResponse()
        try:
            play_url = await self._store_speech_and_get_url(text)
            response.play(play_url)
        except Exception as exc:
            logger.warning("generate_twiml_response falling back to Say: %s", exc)
            response.say(text, voice="Polly.Joanna")
        return str(response)

    async def _store_speech_and_get_url(self, text: str) -> str:
        """Synthesize speech, cache the mp3, and return a public Play URL."""
        if not elevenlabs_key_is_set():
            raise RuntimeError("ElevenLabs not configured")

        AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4().hex}.mp3"
        path = AUDIO_CACHE_DIR / filename
        audio_bytes = get_speaker().generate_speech_sync(text)
        path.write_bytes(audio_bytes)
        return f"{PUBLIC_BASE_URL}/audio/files/{filename}"

    def save_call_recording(
        self,
        call_sid: str,
        recording_url: str,
        from_number: str = "",
        to_number: str = "",
        session_id: str = "",
        status: str = "completed",
        duration: int = 0,
        transcript: str = "",
    ) -> dict[str, Any]:
        """Append call metadata to calls.json for analytics."""
        record = {
            "call_sid": call_sid,
            "from_number": from_number,
            "to_number": to_number or self.phone_number,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "duration": int(duration or 0),
            "recording_url": recording_url or "",
            "status": status,
            "transcript": transcript or "",
        }
        try:
            _append_json_record(CALLS_PATH, record)
        except OSError as exc:
            logger.error("Failed to save call record: %s", exc)
        return record

    def update_call_record(self, call_sid: str, **updates: Any) -> None:
        """Patch the most recent calls.json entry matching call_sid."""
        with _file_lock:
            records = _read_json_list(CALLS_PATH)
            for record in reversed(records):
                if record.get("call_sid") == call_sid or (
                    call_sid and str(record.get("session_id", "")) in call_sid
                ):
                    record.update({k: v for k, v in updates.items() if v is not None})
                    break
            CALLS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with CALLS_PATH.open("w", encoding="utf-8") as handle:
                json.dump(records, handle, indent=2)

    def _finalize_call(
        self,
        session_id: str,
        call_sid: str,
        from_number: str,
        agent_result: dict[str, Any],
        resolved: bool,
    ) -> None:
        """Run post-call automation and refresh the call log entry."""
        session = session_memory.get_session(session_id)
        messages = session_memory.get_full_history(session_id)
        call_data = {
            "call_sid": call_sid or f"session-{session_id}",
            "from_number": from_number,
            "session_id": session_id,
            "messages": messages,
            "intent": agent_result.get("intent", "unknown"),
            "action_taken": agent_result.get("action_taken", "none"),
            "needs_escalation": bool(agent_result.get("needs_escalation")),
            "booking_info": session.get("booking_info", {}),
            "resolved": resolved,
            "duration": max(1, int(session.get("turn_count", 1)) * 45),
            "confidence": float(agent_result.get("confidence", 0.0)),
        }
        try:
            self.post_call.on_call_end(session_id, call_data)
        except Exception as exc:
            logger.error("Post-call automation failed: %s", exc)

        self.save_call_recording(
            call_sid=call_data["call_sid"],
            recording_url="",
            from_number=from_number,
            session_id=session_id,
            status="completed" if resolved else "escalated",
            duration=call_data["duration"],
            transcript="\n".join(
                f"{m.get('role')}: {m.get('content')}" for m in messages
            ),
        )

    async def handle_recording(
        self,
        call_sid: str,
        recording_url: str,
        from_number: str = "",
        duration: int = 0,
    ) -> dict[str, Any]:
        """Download a Twilio recording, transcribe it, and open a ticket."""
        transcript = ""
        try:
            import httpx

            auth = None
            if twilio_configured():
                auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            async with httpx.AsyncClient(timeout=60.0) as client:
                audio_response = await client.get(recording_url + ".mp3", auth=auth)
                audio_response.raise_for_status()
                audio_bytes = audio_response.content
            transcript = get_transcriber().transcribe_file(audio_bytes)
        except Exception as exc:
            logger.error("Voicemail transcription failed: %s", exc)
            transcript = "(transcription unavailable)"

        ticket = self.post_call.create_support_ticket(
            {
                "reason": "Voicemail / recorded call",
                "transcript": transcript,
                "call_sid": call_sid,
                "from_number": from_number,
            }
        )
        self.save_call_recording(
            call_sid=call_sid,
            recording_url=recording_url,
            from_number=from_number,
            status="voicemail",
            duration=duration,
            transcript=transcript,
        )
        return {"status": "ok", "ticket_id": ticket.get("ticket_id"), "transcript": transcript}

    def make_test_call(self, to_number: str) -> dict[str, Any]:
        """Place an outbound test call that plays a short greeting."""
        if not self.client or not self.phone_number:
            raise RuntimeError("Twilio is not configured. Add SID, token, and phone number to .env.")
        if not to_number:
            raise RuntimeError("Provide a destination phone number for the test call.")

        twiml = (
            f'<Response><Say voice="Polly.Joanna">This is a test call from '
            f"{BUSINESS_NAME}. Alex is ready to take patient calls.</Say></Response>"
        )
        call = self.client.calls.create(
            to=to_number,
            from_=self.phone_number,
            twiml=twiml,
        )
        record = self.save_call_recording(
            call_sid=call.sid,
            recording_url="",
            from_number=self.phone_number,
            to_number=to_number,
            status="test",
            duration=0,
        )
        return {"call_sid": call.sid, "status": call.status, "record": record}


_phone_agent: TwilioPhoneAgent | None = None


def get_phone_agent() -> TwilioPhoneAgent:
    """Return a shared TwilioPhoneAgent instance."""
    global _phone_agent
    if _phone_agent is None:
        _phone_agent = TwilioPhoneAgent()
    return _phone_agent
