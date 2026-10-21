"""Deepgram speech-to-text: streaming WebSocket and file transcription."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncGenerator, AsyncIterator, Iterable, Union

from deepgram import AsyncDeepgramClient, DeepgramClient

from ..config import DEEPGRAM_API_KEY, deepgram_key_is_set

logger = logging.getLogger(__name__)

AudioStream = Union[AsyncIterator[bytes], Iterable[bytes]]

# Stop waiting for more audio after this many silent seconds.
SILENCE_TIMEOUT_SECONDS = 30.0


class DeepgramTranscriber:
    """Transcribes audio with Deepgram nova-2."""

    def __init__(self) -> None:
        if not deepgram_key_is_set():
            raise RuntimeError("DEEPGRAM_API_KEY is not set in .env")
        self.api_key = DEEPGRAM_API_KEY
        self.client = DeepgramClient(api_key=self.api_key)
        self.async_client = AsyncDeepgramClient(api_key=self.api_key)
        self.model = "nova-2"

    def transcribe_file(self, audio_bytes: bytes) -> str:
        """Transcribe a complete audio file and return the transcript text.

        Uses the Deepgram sync listen API. Returns an empty string when
        Deepgram finds no speech, and raises RuntimeError on API failures.
        """
        if not audio_bytes:
            logger.warning("transcribe_file called with empty audio")
            return ""

        try:
            response = self.client.listen.v1.media.transcribe_file(
                request=audio_bytes,
                model=self.model,
                smart_format=True,
                punctuate=True,
            )
        except Exception as exc:
            logger.error("Deepgram file transcription failed: %s", exc)
            raise RuntimeError(f"Deepgram transcription failed: {exc}") from exc

        transcript = ""
        confidence = 0.0
        try:
            channel = response.results.channels[0]
            alternative = channel.alternatives[0]
            transcript = (alternative.transcript or "").strip()
            confidence = float(getattr(alternative, "confidence", 0.0) or 0.0)
        except (AttributeError, IndexError, TypeError) as exc:
            logger.warning("Unexpected Deepgram response shape: %s", exc)

        logger.info(
            "Transcribed %d bytes -> %d chars (confidence=%.3f)",
            len(audio_bytes),
            len(transcript),
            confidence,
        )
        return transcript

    def transcribe_file_with_confidence(self, audio_bytes: bytes) -> tuple[str, float]:
        """Like transcribe_file, but also returns the Deepgram confidence score."""
        if not audio_bytes:
            return "", 0.0

        try:
            response = self.client.listen.v1.media.transcribe_file(
                request=audio_bytes,
                model=self.model,
                smart_format=True,
                punctuate=True,
            )
        except Exception as exc:
            logger.error("Deepgram file transcription failed: %s", exc)
            raise RuntimeError(f"Deepgram transcription failed: {exc}") from exc

        transcript = ""
        confidence = 0.0
        try:
            channel = response.results.channels[0]
            alternative = channel.alternatives[0]
            transcript = (alternative.transcript or "").strip()
            confidence = float(getattr(alternative, "confidence", 0.0) or 0.0)
        except (AttributeError, IndexError, TypeError) as exc:
            logger.warning("Unexpected Deepgram response shape: %s", exc)

        logger.info("Transcribed %d bytes", len(audio_bytes))
        return transcript, max(0.0, min(1.0, confidence))

    async def transcribe_stream(self, audio_stream: AudioStream) -> AsyncGenerator[str, None]:
        """Stream audio to Deepgram and yield interim then final transcripts.

        Opens a WebSocket with interim_results enabled. Yields partial
        transcripts as they arrive and a final transcript when Deepgram
        marks the result complete. A 30-second silence timeout ends the
        stream cleanly. On error the generator logs and yields "".
        """
        try:
            async with self.async_client.listen.v1.connect(
                model=self.model,
                interim_results="true",
                smart_format="true",
                punctuate="true",
                utterance_end_ms="1000",
                vad_events="true",
            ) as connection:
                send_task = asyncio.create_task(self._pump_audio(connection, audio_stream))
                last_speech = time.monotonic()

                try:
                    async for message in connection:
                        now = time.monotonic()
                        kind = getattr(message, "type", None) or getattr(
                            message, "event", None
                        )

                        if kind in ("UtteranceEnd", "SpeechStarted"):
                            last_speech = now
                            continue

                        transcript = self._extract_transcript(message)
                        if transcript:
                            last_speech = now
                            yield transcript
                        elif now - last_speech > SILENCE_TIMEOUT_SECONDS:
                            logger.info("Deepgram stream silence timeout reached")
                            break
                finally:
                    send_task.cancel()
                    try:
                        await send_task
                    except asyncio.CancelledError:
                        pass
        except Exception as exc:
            logger.error("Deepgram streaming transcription failed: %s", exc)
            yield ""

    async def _pump_audio(self, connection, audio_stream: AudioStream) -> None:
        """Forward audio chunks into the Deepgram WebSocket."""
        try:
            if hasattr(audio_stream, "__aiter__"):
                async for chunk in audio_stream:  # type: ignore[union-attr]
                    if chunk:
                        await connection.send_media(chunk)
            else:
                for chunk in audio_stream:  # type: ignore[union-attr]
                    if chunk:
                        await connection.send_media(chunk)
            # Signal end-of-stream when the SDK supports it.
            closer = getattr(connection, "send_finalize", None) or getattr(
                connection, "finalize", None
            )
            if callable(closer):
                result = closer()
                if asyncio.iscoroutine(result):
                    await result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Failed while sending audio to Deepgram: %s", exc)

    @staticmethod
    def _extract_transcript(message: object) -> str:
        """Pull transcript text out of a Deepgram streaming message."""
        try:
            channel = getattr(message, "channel", None)
            if channel is None:
                results = getattr(message, "results", None)
                channel = getattr(results, "channel", None) if results else None
            if channel is None:
                return ""
            alternatives = getattr(channel, "alternatives", None) or []
            if not alternatives:
                return ""
            text = (getattr(alternatives[0], "transcript", None) or "").strip()
            return text
        except Exception:
            return ""


_transcriber: DeepgramTranscriber | None = None


def get_transcriber() -> DeepgramTranscriber:
    """Return a shared DeepgramTranscriber, building it on first use."""
    global _transcriber
    if _transcriber is None:
        _transcriber = DeepgramTranscriber()
    return _transcriber
