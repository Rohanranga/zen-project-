"""Shared fixtures for ZenLynx RAG API tests.

Uses temporary directories for Chroma and SQLite so tests are fully
isolated from production data and from each other.  The embedder and
LLM are initialised once per session (model loading is expensive) and
data is ingested once, then shared across all test functions.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session", autouse=True)
def _setup_env():
    """Override storage paths to temp dirs before any app code is imported.

    scope=session ensures the model is loaded only once and the
    ingested data persists across all tests.
    """
    tmpdir = tempfile.mkdtemp(prefix="zenlynx_test_")

    os.environ["CHROMA_PATH"] = str(Path(tmpdir) / "chroma")
    os.environ["SQLITE_PATH"] = str(Path(tmpdir) / "test.db")
    os.environ["EMBEDDING_PROVIDER"] = "local"
    os.environ["LLM_PROVIDER"] = "extractive"

    # DATA_DIR points to the real sample data in the repo.
    project_root = Path(__file__).resolve().parent.parent
    os.environ["DATA_DIR"] = str(project_root / "data")

    # Force settings to reload with test values.
    from app.config import Settings

    import app.config
    app.config.settings = Settings()

    # Ingest sample data so retrieval tests have something to query.
    from app.ingest import run_ingest
    run_ingest(reset=True)

    yield

    # Cleanup is left to the OS for temp dirs — this is test code.


@pytest.fixture(scope="session")
def client(_setup_env) -> TestClient:
    """FastAPI TestClient with all data pre-ingested."""
    from app.main import app

    with TestClient(app) as c:
        yield c
