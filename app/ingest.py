"""CLI and programmatic entry point for document ingestion.

Usage:
    python -m app.ingest              # index all orgs
    python -m app.ingest --org org_001  # index one org
    python -m app.ingest --reset      # drop and rebuild all collections

The ingest pipeline is: load documents → chunk → embed → upsert to Chroma.
It's intentionally synchronous and single-threaded — this is a prototype
where clarity matters more than throughput.
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.chunking import chunk_documents
from app.config import settings
from app.db import init_db, list_orgs, org_exists
from app.embeddings import get_embedder
from app.loaders import load_documents
from app.vectorstore import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def run_ingest(
    org_id: str | None = None,
    reset: bool = False,
) -> dict[str, int]:
    """Run the full ingest pipeline.

    Returns a dict mapping org_id → chunk count for each org processed.
    """
    init_db()
    embedder = get_embedder()
    store = VectorStore(embedder)

    # Determine which orgs to process.
    if org_id:
        if not org_exists(org_id):
            raise ValueError(f"Organization {org_id!r} not found in registry")
        org_ids = [org_id]
    else:
        org_ids = [o["id"] for o in list_orgs()]

    results: dict[str, int] = {}

    for oid in org_ids:
        logger.info("── Ingesting %s ──", oid)

        if reset:
            store.delete_collection(oid)

        try:
            docs = load_documents(oid)
        except FileNotFoundError:
            logger.warning("No data directory for %s — skipping", oid)
            results[oid] = 0
            continue

        if not docs:
            logger.warning("No loadable documents for %s", oid)
            results[oid] = 0
            continue

        chunks = chunk_documents(docs)
        logger.info("  %d documents → %d chunks", len(docs), len(chunks))

        count = store.upsert_chunks(oid, chunks)
        results[oid] = count

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Ingest ESG documents into ChromaDB")
    parser.add_argument("--org", type=str, default=None, help="Ingest a single org")
    parser.add_argument("--reset", action="store_true", help="Drop and rebuild collections")
    args = parser.parse_args()

    try:
        results = run_ingest(org_id=args.org, reset=args.reset)
    except Exception:
        logger.exception("Ingest failed")
        sys.exit(1)

    total = sum(results.values())
    logger.info("Ingest complete — %d total chunks across %d orgs", total, len(results))
    for oid, count in results.items():
        logger.info("  %s: %d chunks", oid, count)


if __name__ == "__main__":
    main()
