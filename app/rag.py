"""RAG pipeline: retrieve → re-rank → build prompt → generate.

The hybrid re-ranking (vector similarity + keyword overlap) exists because
pure vector search struggles with exact-match queries common in ESG
("What is Scope 1?") — adding keyword overlap boosts chunks that contain
the literal terms the user asked about.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

from app.config import settings
from app.embeddings import BaseEmbedder
from app.llm import BaseLLM
from app.schemas import QueryResponse, RetrievedChunk
from app.vectorstore import VectorStore

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Extract lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _keyword_overlap(question: str, text: str) -> float:
    """Jaccard similarity on token multisets — cheap and effective."""
    q_tokens = Counter(_tokenize(question))
    t_tokens = Counter(_tokenize(text))
    intersection = sum((q_tokens & t_tokens).values())
    union = sum((q_tokens | t_tokens).values())
    return intersection / union if union else 0.0


def _hybrid_score(distance: float, keyword_sim: float) -> float:
    """Combine vector distance and keyword overlap into a single score.

    Vector similarity is approximated as 1/(1+distance) to map L2 distance
    into [0,1].  The 0.7/0.3 weighting favours semantic similarity but
    lets exact keyword matches break ties.
    """
    vector_sim = 1.0 / (1.0 + distance)
    return 0.7 * vector_sim + 0.3 * keyword_sim


def retrieve_and_generate(
    *,
    org_id: str,
    question: str,
    vector_store: VectorStore,
    llm: BaseLLM,
    top_k: int = 4,
) -> QueryResponse:
    """Full RAG pipeline for a single question scoped to one organization.

    Steps:
    1. Retrieve ~12 candidate chunks from Chroma.
    2. Re-rank with hybrid score (vector + keyword).
    3. Drop anything above the distance threshold.
    4. Take the top_k results.
    5. Generate an answer from the LLM.
    """
    # ── Step 1: Retrieve candidates ───────────────────────────────
    candidates = vector_store.query(
        org_id=org_id,
        question=question,
        n_results=settings.RETRIEVAL_CANDIDATES,
    )

    if not candidates:
        return QueryResponse(
            organization_id=org_id,
            answer="I cannot find this information in the available documents.",
            sources=[],
            grounded=False,
            retrieved_chunks=[],
        )

    # ── Step 2: Hybrid re-rank ────────────────────────────────────
    for hit in candidates:
        kw_sim = _keyword_overlap(question, hit["text"])
        hit["keyword_sim"] = kw_sim
        hit["score"] = _hybrid_score(hit["distance"], kw_sim)

    candidates.sort(key=lambda h: h["score"], reverse=True)

    # ── Step 3: Distance threshold filter ─────────────────────────
    filtered = [
        h for h in candidates if h["distance"] <= settings.DISTANCE_THRESHOLD
    ]

    if not filtered:
        return QueryResponse(
            organization_id=org_id,
            answer="I cannot find this information in the available documents.",
            sources=[],
            grounded=False,
            retrieved_chunks=[],
        )

    # ── Step 4: Take top_k ────────────────────────────────────────
    top_hits = filtered[:top_k]

    # ── Step 5: Generate answer ───────────────────────────────────
    answer, grounded = llm.generate(question, top_hits)

    # ── Build response ────────────────────────────────────────────
    sources = list(dict.fromkeys(h["document"] for h in top_hits))

    retrieved_chunks = [
        RetrievedChunk(
            document=h["document"],
            chunk_id=h["chunk_id"],
            section=h["section"],
            score=round(h["score"], 4),
            distance=round(h["distance"], 4),
            excerpt=h["text"][:300],
        )
        for h in top_hits
    ]

    return QueryResponse(
        organization_id=org_id,
        answer=answer,
        sources=sources,
        grounded=grounded,
        retrieved_chunks=retrieved_chunks,
    )
