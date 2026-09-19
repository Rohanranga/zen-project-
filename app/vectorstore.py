"""ChromaDB wrapper with three-layer organization data isolation.

Isolation is the headline requirement and is enforced redundantly so that
a bug in any single layer cannot leak data across organizations:

Layer 1 — Physical:  One Chroma collection per org (`org__<id>`).
                     Separate indexes, no shared query path.
Layer 2 — Logical:   Every chunk carries `organization_id` metadata;
                     every query applies `where={"organization_id": id}`.
Layer 3 — Assertion: After retrieval, we verify every hit's metadata
                     matches the requested org. Fail closed on mismatch.
"""

from __future__ import annotations

import logging
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.chunking import Chunk
from app.config import settings
from app.embeddings import BaseEmbedder

logger = logging.getLogger(__name__)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)


class DataIsolationError(Exception):
    """Raised when a retrieved chunk belongs to a different organization.

    This should never happen if layers 1 and 2 are working — its purpose
    is to catch bugs in the isolation logic itself.
    """


def _collection_name(org_id: str) -> str:
    """Deterministic, org-scoped collection name."""
    return f"org__{org_id}"


class VectorStore:
    """Thin wrapper around ChromaDB enforcing per-org isolation."""

    def __init__(self, embedder: BaseEmbedder) -> None:
        self._client = chromadb.PersistentClient(
            path=str(settings.CHROMA_PATH),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._embedder = embedder

    def _get_or_create_collection(self, org_id: str) -> chromadb.Collection:
        """Get (or create) the org's dedicated collection, tagging it with the embedder name."""
        return self._client.get_or_create_collection(
            name=_collection_name(org_id),
            metadata={"embedding_provider": self._embedder.provider_name},
        )

    def _validate_embedder(self, collection: chromadb.Collection) -> None:
        """Refuse to query a collection indexed with a different embedder.

        Mixing embedding spaces silently destroys retrieval quality —
        better to fail loudly than return irrelevant results.
        """
        indexed_provider = (collection.metadata or {}).get("embedding_provider", "")
        if indexed_provider and indexed_provider != self._embedder.provider_name:
            raise ValueError(
                f"Collection was indexed with {indexed_provider!r} but the API is "
                f"running {self._embedder.provider_name!r}. Re-index or change "
                f"EMBEDDING_PROVIDER."
            )

    # ── Write path ────────────────────────────────────────────────

    def upsert_chunks(self, org_id: str, chunks: list[Chunk]) -> int:
        """Index chunks into the org's dedicated collection.

        Returns the number of chunks upserted.
        """
        if not chunks:
            return 0

        collection = self._get_or_create_collection(org_id)

        ids = [c.chunk_id for c in chunks]
        documents = [c.embedded_text for c in chunks]
        metadatas = [
            {
                "organization_id": c.organization_id,
                "document": c.document,
                "section": c.section,
                "chunk_id": c.chunk_id,
            }
            for c in chunks
        ]

        # Batch embed and upsert.
        embeddings = self._embedder.embed(documents)
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Upserted %d chunks for org %s", len(chunks), org_id)
        return len(chunks)

    def delete_collection(self, org_id: str) -> None:
        """Drop an org's entire collection — used during --reset ingest."""
        name = _collection_name(org_id)
        try:
            self._client.delete_collection(name)
            logger.info("Deleted collection %s", name)
        except ValueError:
            logger.debug("Collection %s did not exist", name)

    # ── Read path ─────────────────────────────────────────────────

    def query(
        self,
        org_id: str,
        question: str,
        n_results: int = 12,
    ) -> list[dict]:
        """Retrieve candidate chunks for a question, with full isolation.

        Returns a list of dicts with keys: document, chunk_id, section,
        distance, text.  Results are ordered by distance (ascending).
        """
        name = _collection_name(org_id)
        try:
            collection = self._client.get_collection(name)
        except ValueError:
            return []

        self._validate_embedder(collection)

        if collection.count() == 0:
            return []

        query_embedding = self._embedder.embed([question])[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, collection.count()),
            where={"organization_id": org_id},  # Layer 2: logical filter
            include=["documents", "metadatas", "distances"],
        )

        hits: list[dict] = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]

            # Layer 3: assertion — fail closed if isolation is breached.
            if meta["organization_id"] != org_id:
                raise DataIsolationError(
                    f"Retrieved chunk {meta['chunk_id']} belongs to "
                    f"{meta['organization_id']!r}, expected {org_id!r}. "
                    f"This indicates a critical isolation failure."
                )

            hits.append(
                {
                    "document": meta["document"],
                    "chunk_id": meta["chunk_id"],
                    "section": meta.get("section", ""),
                    "distance": results["distances"][0][i],
                    "text": results["documents"][0][i],
                }
            )

        return hits

    def collection_count(self, org_id: str) -> int:
        """Return the number of chunks indexed for an org (0 if no collection)."""
        try:
            collection = self._client.get_collection(_collection_name(org_id))
            return collection.count()
        except ValueError:
            return 0

    def collection_exists(self, org_id: str) -> bool:
        """Check whether an org has an indexed collection."""
        try:
            self._client.get_collection(_collection_name(org_id))
            return True
        except ValueError:
            return False
