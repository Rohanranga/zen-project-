"""Pydantic v2 request/response models for the API surface.

Org-ID validation is centralised here so every endpoint inherits the same
safety check — regex blocks path traversal and injection before any downstream
code sees the value.
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# ── Org-ID safety ─────────────────────────────────────────────────
_ORG_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def _validate_org_id(value: str) -> str:
    """Reject IDs that could be used for path traversal or collection-name injection."""
    if not _ORG_ID_RE.match(value):
        raise ValueError(
            f"organization_id must match {_ORG_ID_RE.pattern!r}, got {value!r}"
        )
    return value


# ── Request models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Inbound question scoped to a single organization."""

    organization_id: str = Field(..., description="Target organization identifier")
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=20)

    @field_validator("organization_id")
    @classmethod
    def check_org_id(cls, v: str) -> str:
        return _validate_org_id(v)


class IngestRequest(BaseModel):
    """Trigger (re-)indexing, optionally scoped to one org."""

    organization_id: Optional[str] = Field(default=None)
    reset: bool = Field(default=False, description="Drop and rebuild collections")

    @field_validator("organization_id")
    @classmethod
    def check_org_id(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            return _validate_org_id(v)
        return v


# ── Response models ───────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """A single chunk surfaced by the retrieval pipeline."""

    document: str
    chunk_id: str
    section: str
    score: float = Field(description="Hybrid re-rank score (0–1, higher is better)")
    distance: float = Field(description="Raw ChromaDB L2 distance")
    excerpt: str = Field(description="First 300 chars of chunk text")


class QueryResponse(BaseModel):
    """Full answer payload returned by POST /query."""

    organization_id: str
    answer: str
    sources: list[str] = Field(description="De-duplicated source filenames")
    grounded: bool = Field(description="True when the answer is backed by retrieved context")
    retrieved_chunks: list[RetrievedChunk]


class OrganizationInfo(BaseModel):
    """Organization metadata for GET /organizations."""

    id: str
    name: str
    display_name: str
    document_count: int
    chunk_count: int


class HealthResponse(BaseModel):
    """Liveness/readiness probe response."""

    status: str = "ok"
    embedding_provider: str
    llm_provider: str


class IngestResponse(BaseModel):
    """Result of POST /ingest."""

    status: str
    organizations_indexed: list[str]
    total_chunks: int
