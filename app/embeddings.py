"""Pluggable embedding providers.

Two backends — local sentence-transformers (zero API keys, fast iteration)
and OpenAI (higher quality, requires a key).  The provider name is stored
in collection metadata so we can detect and refuse mismatched queries at
runtime rather than silently returning garbage from a different embedding
space.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from app.config import EmbeddingProvider, settings

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Interface that every embedding backend must implement."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identifier stored in Chroma collection metadata."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


class LocalEmbedder(BaseEmbedder):
    """sentence-transformers all-MiniLM-L6-v2 — runs locally, no API key."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(settings.LOCAL_EMBED_MODEL)
        logger.info("Loaded local embedding model: %s", settings.LOCAL_EMBED_MODEL)

    @property
    def provider_name(self) -> str:
        return f"local:{settings.LOCAL_EMBED_MODEL}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI text-embedding-3-small — higher quality, requires OPENAI_API_KEY."""

    def __init__(self) -> None:
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        from openai import OpenAI

        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self._model = settings.OPENAI_EMBED_MODEL
        logger.info("Using OpenAI embedding model: %s", self._model)

    @property
    def provider_name(self) -> str:
        return f"openai:{self._model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        # OpenAI batch limit is 2048 inputs; chunk if needed.
        all_embeddings: list[list[float]] = []
        batch_size = 512
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(input=batch, model=self._model)
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings


def get_embedder() -> BaseEmbedder:
    """Factory — returns the configured embedding provider."""
    if settings.EMBEDDING_PROVIDER == EmbeddingProvider.OPENAI:
        return OpenAIEmbedder()
    return LocalEmbedder()
