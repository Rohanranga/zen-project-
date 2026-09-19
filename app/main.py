"""FastAPI application — the API surface of the ZenLynx RAG service.

Endpoints:
    POST /query          — Ask an ESG question scoped to one organization.
    GET  /organizations  — List orgs with document/chunk counts.
    GET  /health         — Liveness probe with provider info.
    POST /ingest         — Trigger (re-)indexing.

Error semantics:
    404 — organization not in registry.
    422 — malformed input (Pydantic handles this automatically).
    503 — org exists but has no indexed data.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.db import init_db, list_orgs, log_query, org_exists
from app.embeddings import BaseEmbedder, get_embedder
from app.ingest import run_ingest
from app.llm import BaseLLM, get_llm
from app.rag import retrieve_and_generate
from app.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    OrganizationInfo,
    QueryRequest,
    QueryResponse,
)
from app.vectorstore import DataIsolationError, VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Shared state (initialised at startup or on first call) ────────
_embedder: BaseEmbedder | None = None
_llm: BaseLLM | None = None
_store: VectorStore | None = None


def _ensure_resources() -> tuple[VectorStore, BaseLLM, BaseEmbedder]:
    """Ensure DB, embedder, LLM, and vector store are initialised."""
    global _embedder, _llm, _store
    if _embedder is None:
        init_db()
        _embedder = get_embedder()
    if _llm is None:
        _llm = get_llm()
    if _store is None:
        _store = VectorStore(_embedder)
    return _store, _llm, _embedder


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise expensive resources once at startup."""
    store, llm, embedder = _ensure_resources()
    logger.info(
        "Ready — embedder=%s  llm=%s",
        embedder.provider_name,
        llm.provider_name,
    )
    yield


app = FastAPI(
    title="ZenLynx RAG API",
    description="Sustainability/ESG question answering with per-organization data isolation",
    version="0.1.0",
    lifespan=lifespan,
)


# ── POST /query ───────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Answer an ESG question using RAG over one organization's documents."""
    store, llm, _ = _ensure_resources()

    # Fail early on unknown org — don't silently search nothing.
    if not org_exists(request.organization_id):
        raise HTTPException(status_code=404, detail=f"Organization {request.organization_id!r} not found")

    # 503 if the org has no indexed data.
    if not store.collection_exists(request.organization_id):
        raise HTTPException(status_code=503, detail=f"No indexed data for {request.organization_id!r}. Run ingest first.")
    if store.collection_count(request.organization_id) == 0:
        raise HTTPException(status_code=503, detail=f"Index is empty for {request.organization_id!r}. Run ingest first.")

    try:
        response = retrieve_and_generate(
            org_id=request.organization_id,
            question=request.question,
            vector_store=store,
            llm=llm,
            top_k=request.top_k,
        )
    except DataIsolationError as exc:
        logger.error("DATA ISOLATION BREACH: %s", exc)
        raise HTTPException(status_code=500, detail="Data isolation error — request blocked.") from exc

    # Audit log — fire-and-forget.
    log_query(request.organization_id, request.question, response.sources)

    return response


# ── GET /organizations ────────────────────────────────────────────

@app.get("/organizations", response_model=list[OrganizationInfo])
def organizations() -> list[OrganizationInfo]:
    """List all registered organizations with document/chunk counts."""
    store, _, _ = _ensure_resources()
    orgs = list_orgs()
    result: list[OrganizationInfo] = []
    for org in orgs:
        chunk_count = store.collection_count(org["id"])
        # Approximate doc count from data directory.
        org_dir = settings.DATA_DIR / org["id"]
        doc_count = 0
        if org_dir.is_dir():
            doc_count = sum(
                1 for f in org_dir.iterdir()
                if f.suffix.lower() in {".md", ".txt", ".pdf"}
            )
        result.append(
            OrganizationInfo(
                id=org["id"],
                name=org["name"],
                display_name=org["display_name"],
                document_count=doc_count,
                chunk_count=chunk_count,
            )
        )
    return result


# ── GET /health ───────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe — reports active providers."""
    _, llm, embedder = _ensure_resources()
    return HealthResponse(
        status="ok",
        embedding_provider=embedder.provider_name,
        llm_provider=llm.provider_name,
    )


# ── POST /ingest ──────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    """Re-index documents. Optionally scoped to a single organization."""
    global _store

    if request.organization_id and not org_exists(request.organization_id):
        raise HTTPException(status_code=404, detail=f"Organization {request.organization_id!r} not found")

    results = run_ingest(org_id=request.organization_id, reset=request.reset)

    # Refresh the store reference after potential collection recreation.
    _, _, embedder = _ensure_resources()
    _store = VectorStore(embedder)

    return IngestResponse(
        status="success",
        organizations_indexed=list(results.keys()),
        total_chunks=sum(results.values()),
    )
