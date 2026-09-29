"""Load runtime settings from environment variables and the project .env file."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# backend/config.py lives one level below the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _resolve_path(value: str) -> Path:
    """Resolve a relative path against the project root.

    Absolute paths are returned unchanged so a machine-specific
    CHROMA_PERSIST_DIR still works.
    """
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


class Settings:
    """Typed access to every environment variable the backend reads."""

    def __init__(self) -> None:
        self.groq_api_key: str = os.getenv("GROQ_API_KEY", "").strip()
        self.business_name: str = os.getenv("BUSINESS_NAME", "City Medical Clinic").strip()
        self.chroma_persist_dir: Path = _resolve_path(
            os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        )
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO").strip().upper()
        self.knowledge_base_dir: Path = PROJECT_ROOT / "data" / "knowledge_base"
        self.bm25_index_path: Path = self.chroma_persist_dir / "bm25_index.pkl"
        self.embedding_model_name: str = "all-MiniLM-L6-v2"
        self.chunk_size: int = 500
        self.chunk_overlap: int = 50
        self.collection_name: str = "knowledge_base"

        # Groq has retired the Llama 3.1 8B endpoints. gpt-oss-20b is the
        # current small, fast chat model. Override with GROQ_MODEL.
        self.groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
        self.intent_temperature: float = 0.1
        self.response_temperature: float = 0.3
        self.max_output_tokens: int = 512

        # Retrieval below this cosine similarity is treated as "not in the
        # knowledge base" and hands the caller to a human.
        self.escalation_confidence_threshold: float = 0.4
        self.retrieval_top_k: int = 5

        # Sliding window and compression settings for session memory.
        self.context_window_messages: int = 6
        self.summarize_after_turns: int = 20

        self.appointments_path: Path = PROJECT_ROOT / "appointments.json"
        self.escalations_path: Path = PROJECT_ROOT / "escalations.json"

    @property
    def system_prompt(self) -> str:
        """The agent persona shared by every LLM call that speaks to a caller."""
        return (
            f"You are Alex, a helpful customer service agent for {self.business_name}. "
            "You speak naturally and concisely, like a real receptionist would on the phone. "
            "Always be warm but efficient. If you cannot find information in your knowledge base "
            "with high confidence, say so honestly rather than guessing. "
            "When booking appointments, confirm all details before finalizing."
        )

    def configure_logging(self) -> None:
        """Apply LOG_LEVEL to the root logger.

        basicConfig is a no-op once handlers exist, so repeated calls
        during tests or re-ingestion do not stack formatters.
        """
        level = getattr(logging, self.log_level, logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
        logging.getLogger().setLevel(level)


settings = Settings()
