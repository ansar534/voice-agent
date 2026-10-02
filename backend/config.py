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

# Groq retired the llama-3.1-8b endpoints; gpt-oss-20b is the current
# small, fast chat model. Override with GROQ_MODEL in .env.
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()

# --- Business --------------------------------------------------------------
BUSINESS_NAME: str = os.getenv("BUSINESS_NAME", "City Medical Clinic").strip()

# --- Storage paths ---------------------------------------------------------
CHROMA_PERSIST_DIR: Path = _resolve(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
KNOWLEDGE_BASE_DIR: Path = _resolve(os.getenv("KNOWLEDGE_BASE_DIR", "./data/knowledge_base"))
BM25_INDEX_PATH: Path = _resolve(os.getenv("BM25_INDEX_PATH", "./bm25_index.pkl"))
APPOINTMENTS_PATH: Path = _resolve(os.getenv("APPOINTMENTS_PATH", "./appointments.json"))
ESCALATIONS_PATH: Path = _resolve(os.getenv("ESCALATIONS_PATH", "./escalations.json"))

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


def groq_key_is_set() -> bool:
    """Report whether a usable Groq key is present.

    Treats the placeholder shipped in .env.example as "not set" so the
    dashboard can warn before the first call fails.
    """
    return bool(GROQ_API_KEY) and GROQ_API_KEY != "your_groq_api_key_here"
