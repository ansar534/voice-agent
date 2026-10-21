"""ElevenLabs text-to-speech: full audio and streaming chunk generators."""

from __future__ import annotations

import logging
import re
from typing import AsyncGenerator, Generator, Iterator

from elevenlabs.client import ElevenLabs

from ..config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, elevenlabs_key_is_set

logger = logging.getLogger(__name__)

# Friendly names mapped to published ElevenLabs voice IDs.
VOICE_NAME_TO_ID = {
    "rachel": "21m00Tcm4TlvDq8ikWAM",
    "domi": "AZnzlk1XvdvUeBnXmlld",
    "bella": "EXAVITQu4vr4xnSDxMaL",
    "antoni": "ErXwobaYiN019PkySvjV",
    "elli": "MF3mGyEYCl7XYWbV9V6O",
    "josh": "TxGEqnHWrfWFTfGW9XjX",
    "arnold": "VR6AewLTigWG4xSOukaG",
    "adam": "pNInz6obpgDQGcFmaJgB",
    "sam": "yoZ06aMxZJJ28mfd3POQ",
}

ELEVENLABS_MODEL = "eleven_monolingual_v1"
MAX_CHARS_PER_REQUEST = 1000


class ElevenLabsSpeaker:
    """Converts agent text replies into spoken mp3 audio."""

    def __init__(self) -> None:
        if not elevenlabs_key_is_set():
            raise RuntimeError("ELEVENLABS_API_KEY is not set in .env")
        self.client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        self.voice_id = self._resolve_voice_id(ELEVENLABS_VOICE_ID)
        self.model_id = ELEVENLABS_MODEL

    @staticmethod
    def _resolve_voice_id(configured: str) -> str:
        """Accept either a raw voice id or a known display name like Rachel."""
        value = (configured or "Rachel").strip()
        mapped = VOICE_NAME_TO_ID.get(value.lower())
        return mapped or value

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split long text into sentence-sized pieces under the char limit."""
        cleaned = " ".join(text.split())
        if not cleaned:
            return []
        if len(cleaned) <= MAX_CHARS_PER_REQUEST:
            return [cleaned]

        pieces = re.split(r"(?<=[.!?])\s+", cleaned)
        batches: list[str] = []
        current = ""
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if len(piece) > MAX_CHARS_PER_REQUEST:
                if current:
                    batches.append(current)
                    current = ""
                for start in range(0, len(piece), MAX_CHARS_PER_REQUEST):
                    batches.append(piece[start : start + MAX_CHARS_PER_REQUEST])
                continue
            candidate = f"{current} {piece}".strip() if current else piece
            if len(candidate) <= MAX_CHARS_PER_REQUEST:
                current = candidate
            else:
                batches.append(current)
                current = piece
        if current:
            batches.append(current)
        return batches

    def _synthesize(self, text: str) -> bytes:
        """Call ElevenLabs convert() and collect the returned byte iterator."""
        stream: Iterator[bytes] = self.client.text_to_speech.convert(
            voice_id=self.voice_id,
            text=text,
            model_id=self.model_id,
            output_format="mp3_44100_128",
        )
        return b"".join(chunk for chunk in stream if chunk)

    async def generate_speech(self, text: str) -> bytes:
        """Convert text to speech and return complete mp3 audio bytes.

        Text longer than 1000 characters is split on sentence boundaries,
        each piece is synthesized, and the mp3 payloads are concatenated.
        """
        parts = self._split_sentences(text)
        if not parts:
            return b""

        audio_chunks: list[bytes] = []
        for part in parts:
            try:
                audio_chunks.append(self._synthesize(part))
            except Exception as exc:
                logger.error("ElevenLabs speech generation failed: %s", exc)
                raise RuntimeError(f"ElevenLabs speech generation failed: {exc}") from exc

        audio_bytes = b"".join(audio_chunks)
        logger.info(
            "Generated speech: %d chars → %d bytes",
            len(text),
            len(audio_bytes),
        )
        return audio_bytes

    def generate_speech_sync(self, text: str) -> bytes:
        """Synchronous wrapper used by FastAPI worker threads and Twilio."""
        parts = self._split_sentences(text)
        if not parts:
            return b""

        audio_chunks: list[bytes] = []
        for part in parts:
            try:
                audio_chunks.append(self._synthesize(part))
            except Exception as exc:
                logger.error("ElevenLabs speech generation failed: %s", exc)
                raise RuntimeError(f"ElevenLabs speech generation failed: {exc}") from exc

        audio_bytes = b"".join(audio_chunks)
        logger.info(
            "Generated speech: %d chars → %d bytes",
            len(text),
            len(audio_bytes),
        )
        return audio_bytes

    def generate_speech_stream(self, text: str) -> Generator[bytes, None, None]:
        """Yield mp3 audio chunks as they arrive from ElevenLabs.

        Long text is split into sentences; each sentence is streamed so
        the caller can start playback before the whole reply is ready.
        """
        for part in self._split_sentences(text):
            try:
                stream = self.client.text_to_speech.stream(
                    voice_id=self.voice_id,
                    text=part,
                    model_id=self.model_id,
                    output_format="mp3_44100_128",
                )
                for chunk in stream:
                    if chunk:
                        yield chunk
            except Exception as exc:
                logger.error("ElevenLabs streaming speech failed: %s", exc)
                raise RuntimeError(f"ElevenLabs streaming speech failed: {exc}") from exc

    async def generate_speech_stream_async(self, text: str) -> AsyncGenerator[bytes, None]:
        """Async wrapper around generate_speech_stream for async callers."""
        for chunk in self.generate_speech_stream(text):
            yield chunk


_speaker: ElevenLabsSpeaker | None = None


def get_speaker() -> ElevenLabsSpeaker:
    """Return a shared ElevenLabsSpeaker, building it on first use."""
    global _speaker
    if _speaker is None:
        _speaker = ElevenLabsSpeaker()
    return _speaker
