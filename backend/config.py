"""Central configuration loaded from the project .env file.

Every other module imports its settings from here, so no API key or path
is ever hardcoded elsewhere in the codebase.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# backend/config.py sits one level below the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _resolve(value: str) -> Path:
    """Turn a configured path into an absolute one.

    Relative values are resolved against the project root so the app
    behaves the same no matter which directory it is launched from.
    """
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


# --- Credentials -----------------------------------------------------------
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "").strip()
ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "Rachel").strip() or "Rachel"
TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
TWILIO_PHONE_NUMBER: str = os.getenv("TWILIO_PHONE_NUMBER", "").strip()

# Groq retired the llama-3.1-8b endpoints; gpt-oss-20b is the current
# small, fast chat model. Override with GROQ_MODEL in .env.
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()

# --- Business --------------------------------------------------------------
BUSINESS_NAME: str = os.getenv("BUSINESS_NAME", "City Medical Clinic").strip()
CLINIC_ADDRESS: str = os.getenv(
    "CLINIC_ADDRESS",
    "123 Health Avenue, Suite 100",
).strip()
CLINIC_STAFF_EMAIL: str = os.getenv("CLINIC_STAFF_EMAIL", "").strip()
SMTP_HOST: str = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER: str = os.getenv("SMTP_USER", "").strip()
SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "").strip()
SMTP_FROM: str = os.getenv("SMTP_FROM", SMTP_USER or "noreply@citymedical.local").strip()
PUBLIC_BASE_URL: str = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")

# --- Storage paths ---------------------------------------------------------
CHROMA_PERSIST_DIR: Path = _resolve(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
KNOWLEDGE_BASE_DIR: Path = _resolve(os.getenv("KNOWLEDGE_BASE_DIR", "./data/knowledge_base"))
BM25_INDEX_PATH: Path = _resolve(os.getenv("BM25_INDEX_PATH", "./bm25_index.pkl"))
APPOINTMENTS_PATH: Path = _resolve(os.getenv("APPOINTMENTS_PATH", "./appointments.json"))
ESCALATIONS_PATH: Path = _resolve(os.getenv("ESCALATIONS_PATH", "./escalations.json"))
CALLS_PATH: Path = _resolve(os.getenv("CALLS_PATH", "./calls.json"))
CALL_ANALYTICS_PATH: Path = _resolve(os.getenv("CALL_ANALYTICS_PATH", "./call_analytics.json"))
SUPPORT_TICKETS_PATH: Path = _resolve(os.getenv("SUPPORT_TICKETS_PATH", "./support_tickets.json"))
AUDIO_CACHE_DIR: Path = _resolve(os.getenv("AUDIO_CACHE_DIR", "./audio_cache"))
PHONE_SETTINGS_PATH: Path = _resolve(os.getenv("PHONE_SETTINGS_PATH", "./phone_settings.json"))

# --- Retrieval and agent behaviour ----------------------------------------
COLLECTION_NAME: str = "knowledge_base"
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50

# Retrieval weaker than this is treated as "not in the knowledge base"
# and hands the caller to a human instead of guessing.
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))

# Semantic and keyword weights used to merge hybrid search results.
SEMANTIC_WEIGHT: float = 0.6
BM25_WEIGHT: float = 0.4

# How many past messages are replayed to the LLM each turn.
MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", "6"))

# Temperatures: near-deterministic for routing, slightly warm for speech.
INTENT_TEMPERATURE: float = 0.1
RESPONSE_TEMPERATURE: float = 0.3
MAX_OUTPUT_TOKENS: int = 512

# --- Cost estimates used by the financial analytics tab -------------------
COST_PER_GROQ_1K_TOKENS: float = float(os.getenv("COST_PER_GROQ_1K_TOKENS", "0.0002"))
COST_PER_DEEPGRAM_MINUTE: float = float(os.getenv("COST_PER_DEEPGRAM_MINUTE", "0.0043"))
COST_PER_ELEVENLABS_1K_CHARS: float = float(os.getenv("COST_PER_ELEVENLABS_1K_CHARS", "0.18"))
COST_PER_TWILIO_MINUTE: float = float(os.getenv("COST_PER_TWILIO_MINUTE", "0.014"))
COST_PER_HUMAN_AGENT_CALL: float = float(os.getenv("COST_PER_HUMAN_AGENT_CALL", "8.0"))

# --- Logging ---------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").strip().upper()

SYSTEM_PROMPT: str = (
    f"You are Alex, a warm and efficient medical receptionist at {BUSINESS_NAME}. "
    "Answer using ONLY the information in the context provided. "
    "If the context does not contain the answer, say: "
    "'I don't have that information - let me connect you with our staff.' "
    "Never make up medical information. Be concise and natural, like a real receptionist."
)


def configure_logging() -> None:
    """Apply LOG_LEVEL to the root logger.

    Safe to call more than once; basicConfig is a no-op once a handler
    exists, so repeated calls do not stack duplicate formatters.
    """
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logging.getLogger().setLevel(level)


def _key_is_set(value: str, placeholder: str = "") -> bool:
    """True when a credential is present and is not an obvious placeholder."""
    if not value:
        return False
    lowered = value.lower()
    if placeholder and lowered == placeholder.lower():
        return False
    if lowered.startswith("your_") or lowered.endswith("_here"):
        return False
    return True


def groq_key_is_set() -> bool:
    """Report whether a usable Groq key is present."""
    return _key_is_set(GROQ_API_KEY, "your_groq_api_key_here")


def deepgram_key_is_set() -> bool:
    """Report whether a usable Deepgram key is present."""
    return _key_is_set(DEEPGRAM_API_KEY, "your_deepgram_api_key_here")


def elevenlabs_key_is_set() -> bool:
    """Report whether a usable ElevenLabs key is present."""
    return _key_is_set(ELEVENLABS_API_KEY, "your_elevenlabs_api_key_here")


def twilio_configured() -> bool:
    """Report whether Twilio SID, token, and phone number are all set."""
    return (
        _key_is_set(TWILIO_ACCOUNT_SID, "your_twilio_account_sid_here")
        and _key_is_set(TWILIO_AUTH_TOKEN, "your_twilio_auth_token_here")
        and bool(TWILIO_PHONE_NUMBER)
    )
