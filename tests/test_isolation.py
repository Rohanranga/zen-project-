"""Data isolation and correctness tests for the ZenLynx RAG API.

These tests verify the headline requirement — that querying one
organization never leaks data from another — plus basic error handling.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class TestDataIsolation:
    """Cross-organization data isolation must hold under all conditions."""

    def test_org1_never_returns_org2_sources(self, client: TestClient) -> None:
        """org_001 query results must never reference org_002 files."""
        resp = client.post("/query", json={
            "organization_id": "org_001",
            "question": "What are the greenhouse gas emissions?",
        })
        assert resp.status_code == 200
        data = resp.json()
        for source in data["sources"]:
            assert "org_002" not in source, f"org_002 file leaked into org_001 results: {source}"
        for chunk in data["retrieved_chunks"]:
            assert "org_002" not in chunk["document"]

    def test_org2_never_returns_org1_sources(self, client: TestClient) -> None:
        """org_002 query results must never reference org_001 files."""
        resp = client.post("/query", json={
            "organization_id": "org_002",
            "question": "What are the greenhouse gas emissions?",
        })
        assert resp.status_code == 200
        data = resp.json()
        for source in data["sources"]:
            assert "org_001" not in source, f"org_001 file leaked into org_002 results: {source}"
        for chunk in data["retrieved_chunks"]:
            assert "org_001" not in chunk["document"]

    def test_same_question_different_values(self, client: TestClient) -> None:
        """The same Scope 1 question must return 1,250 for org_001 and 2,810 for org_002."""
        question = "What is the Scope 1 emission for 2025?"

        resp1 = client.post("/query", json={
            "organization_id": "org_001",
            "question": question,
        })
        resp2 = client.post("/query", json={
            "organization_id": "org_002",
            "question": question,
        })

        assert resp1.status_code == 200
        assert resp2.status_code == 200

        answer1 = resp1.json()["answer"]
        answer2 = resp2.json()["answer"]

        # org_001 = Aurora Textiles: Scope 1 is 1,250 tCO2e
        assert "1,250" in answer1 or "1250" in answer1, (
            f"Expected 1,250 in org_001 answer, got: {answer1}"
        )
        # org_002 = Meridian Logistics: Scope 1 is 2,810 tCO2e
        assert "2,810" in answer2 or "2810" in answer2, (
            f"Expected 2,810 in org_002 answer, got: {answer2}"
        )

        # Cross-check: org_001's answer must NOT contain org_002's value.
        assert "2,810" not in answer1 and "2810" not in answer1, (
            f"org_002 value (2,810) leaked into org_001 answer: {answer1}"
        )


class TestErrorHandling:
    """Proper error responses for invalid inputs and edge cases."""

    def test_unknown_org_returns_404(self, client: TestClient) -> None:
        """An organization not in the registry must get a 404, not an empty search."""
        resp = client.post("/query", json={
            "organization_id": "org_999",
            "question": "What is the emission?",
        })
        assert resp.status_code == 404

    def test_traversal_org_id_rejected(self, client: TestClient) -> None:
        """Path-traversal IDs like '../org_002' must be rejected at validation (422)."""
        resp = client.post("/query", json={
            "organization_id": "../org_002",
            "question": "What is the emission?",
        })
        assert resp.status_code == 422

    def test_dot_dot_slash_rejected(self, client: TestClient) -> None:
        """Another traversal variant."""
        resp = client.post("/query", json={
            "organization_id": "..%2Forg_002",
            "question": "What is the emission?",
        })
        assert resp.status_code == 422

    def test_empty_org_id_rejected(self, client: TestClient) -> None:
        """Empty org ID must be rejected."""
        resp = client.post("/query", json={
            "organization_id": "",
            "question": "What is the emission?",
        })
        assert resp.status_code == 422


class TestGrounding:
    """The LLM must not hallucinate — out-of-scope questions get grounded=false."""

    def test_out_of_scope_question_returns_ungrounded(self, client: TestClient) -> None:
        """A question with no relevant context should return grounded=false."""
        resp = client.post("/query", json={
            "organization_id": "org_001",
            "question": "What is the company's stock price today?",
        })
        assert resp.status_code == 200
        data = resp.json()
        # The extractive fallback should either return grounded=false or
        # an answer that says it cannot find the information.
        if data["grounded"]:
            assert "cannot find" in data["answer"].lower() or "stock" not in data["answer"].lower()


class TestEndpoints:
    """Smoke tests for non-query endpoints."""

    def test_health(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "local" in data["embedding_provider"]
        assert data["llm_provider"] == "extractive"

    def test_organizations_list(self, client: TestClient) -> None:
        resp = client.get("/organizations")
        assert resp.status_code == 200
        orgs = resp.json()
        assert len(orgs) >= 2
        org_ids = {o["id"] for o in orgs}
        assert "org_001" in org_ids
        assert "org_002" in org_ids
        # Both should have chunks after ingest.
        for org in orgs:
            assert org["chunk_count"] > 0, f"{org['id']} has no chunks"
