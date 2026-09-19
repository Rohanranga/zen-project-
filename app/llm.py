"""Pluggable LLM providers: extractive fallback, OpenAI, and Anthropic.

The extractive fallback is the key design decision — it lets the entire
pipeline run end-to-end with zero API keys.  Rather than generating text,
it finds the most relevant chunk via keyword overlap and returns it
verbatim.  This is "good enough" for demos and testing while making the
prototype self-contained.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

from app.config import LLMProvider, settings

logger = logging.getLogger(__name__)

# ── Shared prompt template ────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are a sustainability/ESG analyst assistant. "
    "Answer the user's question ONLY using the numbered context passages below. "
    "If the answer is not contained in the context, say exactly: "
    '"I cannot find this information in the available documents." '
    "Never use general knowledge. Cite passage numbers in your answer."
)


def _build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the LLM."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("document", "unknown")
        section = chunk.get("section", "")
        text = chunk.get("text", "")
        header = f"[{i}] Source: {source}"
        if section:
            header += f" | Section: {section}"
        parts.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(parts)


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens for keyword overlap scoring."""
    return re.findall(r"[a-z0-9]+", text.lower())


# ── Base class ────────────────────────────────────────────────────

class BaseLLM(ABC):
    """Interface for all LLM backends."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier."""

    @abstractmethod
    def generate(self, question: str, chunks: list[dict]) -> tuple[str, bool]:
        """Return (answer_text, is_grounded)."""


_STOPWORDS = {
    "a", "about", "above", "after", "again", "all", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "did", "do", "does", "doing", "down", "during",
    "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "would", "you", "your", "yours",
}


# ── Extractive fallback (no API key) ─────────────────────────────

class ExtractiveLLM(BaseLLM):
    """Keyword-based extractive answerer — no API key required.

    Selects the chunk with the highest keyword overlap with the question
    and returns it verbatim. Grounded is True only when a reasonably
    relevant chunk exists.
    """

    @property
    def provider_name(self) -> str:
        return "extractive"

    def generate(self, question: str, chunks: list[dict]) -> tuple[str, bool]:
        if not chunks:
            return "I cannot find this information in the available documents.", False

        content_tokens = [t for t in _tokenize(question) if t not in _STOPWORDS]
        if not content_tokens:
            content_tokens = _tokenize(question)

        best_chunk = chunks[0]
        best_coverage = 0.0

        for chunk in chunks:
            chunk_tokens = set(_tokenize(chunk.get("text", "") + " " + chunk.get("section", "")))
            matching = [t for t in content_tokens if t in chunk_tokens]
            coverage = len(matching) / len(content_tokens) if content_tokens else 0.0
            if coverage > best_coverage:
                best_coverage = coverage
                best_chunk = chunk

        if best_coverage < 0.4:
            return "I cannot find this information in the available documents.", False

        text = best_chunk.get("text", "").strip()
        source = best_chunk.get("document", "unknown")
        section = best_chunk.get("section", "")

        header = f"From **{source}**"
        if section:
            header += f" ({section})"
            if text.startswith(section):
                text = text[len(section):].strip()

        return f"{header}:\n\n{text}", True


# ── OpenAI ────────────────────────────────────────────────────────

class OpenAILLM(BaseLLM):
    """OpenAI GPT-4o-mini with strict grounding instructions."""

    def __init__(self) -> None:
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        from openai import OpenAI

        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self._model = settings.OPENAI_LLM_MODEL

    @property
    def provider_name(self) -> str:
        return f"openai:{self._model}"

    def generate(self, question: str, chunks: list[dict]) -> tuple[str, bool]:
        if not chunks:
            return "I cannot find this information in the available documents.", False

        context = _build_context_block(chunks)
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
        )
        answer = response.choices[0].message.content or ""
        grounded = "cannot find" not in answer.lower()
        return answer.strip(), grounded


# ── Anthropic ─────────────────────────────────────────────────────

class AnthropicLLM(BaseLLM):
    """Anthropic Claude with strict grounding instructions."""

    def __init__(self) -> None:
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        from anthropic import Anthropic

        self._client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self._model = settings.ANTHROPIC_LLM_MODEL

    @property
    def provider_name(self) -> str:
        return f"anthropic:{self._model}"

    def generate(self, question: str, chunks: list[dict]) -> tuple[str, bool]:
        if not chunks:
            return "I cannot find this information in the available documents.", False

        context = _build_context_block(chunks)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                }
            ],
        )
        answer = response.content[0].text
        grounded = "cannot find" not in answer.lower()
        return answer.strip(), grounded


# ── OmniRoute (OpenAI-compatible gateway) ─────────────────────────

class OmniRouteLLM(BaseLLM):
    """Any OpenAI-compatible model served by an OmniRoute gateway.

    OmniRoute exposes an OpenAI-compatible API (typically at
    ``http://localhost:<port>/v1``) that proxies to hosted models by name.
    This lets the RAG API generate answers through the user's configured
    gateway without depending on the vendor SDK details — the ``openai``
    SDK is used HTTP-compatibly against the gateway's ``/v1`` endpoint.
    """

    def __init__(self) -> None:
        if not settings.OMNIROUTE_API_KEY:
            raise ValueError("OMNIROUTE_API_KEY is required when LLM_PROVIDER=omniroute")
        if not settings.OMNIROUTE_MODEL:
            raise ValueError("OMNIROUTE_MODEL is required when LLM_PROVIDER=omniroute")
        from openai import OpenAI

        self._client = OpenAI(
            api_key=settings.OMNIROUTE_API_KEY,
            base_url=settings.OMNIROUTE_BASE_URL,
        )
        self._model = settings.OMNIROUTE_MODEL

    @property
    def provider_name(self) -> str:
        return f"omniroute:{self._model}"

    def generate(self, question: str, chunks: list[dict]) -> tuple[str, bool]:
        if not chunks:
            return "I cannot find this information in the available documents.", False

        context = _build_context_block(chunks)
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
        )
        answer = response.choices[0].message.content or ""
        grounded = "cannot find" not in answer.lower()
        return answer.strip(), grounded


# ── Factory ───────────────────────────────────────────────────────

def get_llm() -> BaseLLM:
    """Return the configured LLM provider."""
    if settings.LLM_PROVIDER == LLMProvider.OPENAI:
        return OpenAILLM()
    if settings.LLM_PROVIDER == LLMProvider.ANTHROPIC:
        return AnthropicLLM()
    if settings.LLM_PROVIDER == LLMProvider.OMNIROUTE:
        return OmniRouteLLM()
    return ExtractiveLLM()
