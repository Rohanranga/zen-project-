"""Heading-aware text chunker for ESG/sustainability documents.

ESG reports are full of context-dependent numbers ("Scope 1 (direct):
1,250 tCO2e") that become meaningless once separated from their section
heading.  This chunker tracks the nearest markdown heading and prefixes
it to each chunk's embedded text, so the embedding model sees
"## Greenhouse Gas Emissions — Scope 1 (direct): 1,250 tCO2e" rather
than just "1,250 tCO2e".

Splitting on blank lines (paragraph boundaries) rather than a fixed
character count preserves semantic units — a paragraph about water
consumption stays in one chunk rather than being cut mid-sentence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import settings
from app.loaders import RawDocument


@dataclass(frozen=True)
class Chunk:
    """A chunk of text with full provenance for downstream retrieval."""

    chunk_id: str
    text: str
    section: str
    document: str
    organization_id: str

    @property
    def embedded_text(self) -> str:
        """The text sent to the embedding model — heading + body."""
        if self.section and not self.text.startswith(self.section):
            return f"{self.section}\n{self.text}"
        return self.text


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines, filtering out empty fragments."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def chunk_document(doc: RawDocument) -> list[Chunk]:
    """Split a document into heading-aware, overlapping chunks.

    Algorithm:
    1. Split text into paragraphs (blank-line delimited).
    2. Track the active markdown heading.
    3. When a new heading appears, flush the previous section's accumulated text
       so section boundaries are respected.
    4. Greedily pack paragraphs into chunks of ~CHUNK_SIZE chars with ~CHUNK_OVERLAP overlap.
    """
    max_size: int = settings.CHUNK_SIZE
    overlap: int = settings.CHUNK_OVERLAP

    paragraphs = _split_paragraphs(doc.text)
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    active_section = ""
    current_parts: list[str] = []
    current_len = 0
    chunk_index = 0

    def _emit(section: str) -> None:
        """Flush accumulated paragraphs into a Chunk."""
        nonlocal chunk_index
        if not current_parts:
            return
        body = "\n\n".join(current_parts)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.filename}::chunk_{chunk_index:04d}",
                text=body,
                section=section,
                document=doc.filename,
                organization_id=doc.organization_id,
            )
        )
        chunk_index += 1

    for para in paragraphs:
        heading_match = _HEADING_RE.match(para)
        if heading_match:
            # When entering a new heading, flush previous section content if non-empty
            if current_parts:
                _emit(active_section)
                current_parts = []
                current_len = 0
            active_section = heading_match.group(0)

        added_len = len(para) + (2 if current_parts else 0)
        if current_len + added_len > max_size and current_parts:
            _emit(active_section)

            # Overlap: carry the tail into the new chunk for continuity
            overlap_text = "\n\n".join(current_parts)
            if len(overlap_text) > overlap:
                overlap_text = overlap_text[-overlap:]
            current_parts = [overlap_text]
            current_len = len(overlap_text)

        current_parts.append(para)
        current_len += added_len

    _emit(active_section)
    return chunks


def chunk_documents(docs: list[RawDocument]) -> list[Chunk]:
    """Chunk a batch of documents, preserving document and org provenance."""
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks
