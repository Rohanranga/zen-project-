"""Centralised configuration — every tunable lives here.

Uses pydantic-settings so values come from .env or environment variables.
Validated on import so misconfiguration fails fast rather than mid-request.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingProvider(str, Enum):
    """Supported embedding backends."""
    LOCAL = "local"
    OPENAI = "openai"


class LLMProvider(str, Enum):
    """Supported LLM backends."""
    EXTRACTIVE = "extractive"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OMNIROUTE = "omniroute"


class Settings(BaseSettings):
    """All runtime settings, read once at startup."""

    # ── Providers ──────────────────────────────────────────────────
    EMBEDDING_PROVIDER: EmbeddingProvider = EmbeddingProvider.LOCAL
    LLM_PROVIDER: LLMProvider = LLMProvider.EXTRACTIVE

    # ── API keys (only validated when the matching provider is active) ─
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    OMNIROUTE_API_KEY: str = ""
    OMNIROUTE_BASE_URL: str = "http://localhost:20128/v1"
    OMNIROUTE_MODEL: str = ""

    # ── Storage ────────────────────────────────────────────────────
    CHROMA_PATH: Path = Path("chroma_store")
    SQLITE_PATH: Path = Path("zenlynx.db")
    DATA_DIR: Path = Path("data")

    # ── Retrieval tuning ───────────────────────────────────────────
    DEFAULT_TOP_K: int = 4
    RETRIEVAL_CANDIDATES: int = 12
    DISTANCE_THRESHOLD: float = 1.2
    CHUNK_SIZE: int = 900
    CHUNK_OVERLAP: int = 150

    # ── Model names ────────────────────────────────────────────────
    LOCAL_EMBED_MODEL: str = "all-MiniLM-L6-v2"
    OPENAI_EMBED_MODEL: str = "text-embedding-3-small"
    OPENAI_LLM_MODEL: str = "gpt-4o-mini"
    ANTHROPIC_LLM_MODEL: str = "claude-3-5-haiku-20241022"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Module-level singleton — import this everywhere.
settings = Settings()
