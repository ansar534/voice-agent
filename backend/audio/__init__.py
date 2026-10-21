"""Speech-to-text and text-to-speech adapters."""

from .speaker import ElevenLabsSpeaker
from .transcriber import DeepgramTranscriber

__all__ = ["DeepgramTranscriber", "ElevenLabsSpeaker"]
