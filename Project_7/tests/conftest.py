"""Shared pytest fixtures for Project 7."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app import dependencies  # noqa: E402
from app.main import app  # noqa: E402
from app.services.rag_service import AdvancedRagService  # noqa: E402


@pytest.fixture
def rag_service(tmp_path) -> AdvancedRagService:
    return AdvancedRagService(
        storage_file=str(tmp_path / "corpus.json"),
        embedding_dimensions=128,
        chunk_size=80,
        chunk_overlap=10,
        bm25_k1=1.5,
        bm25_b=0.75,
        rrf_k=60,
        reranker_enabled=True,
        answer_backend="rule_based",
    )


@pytest.fixture
def client(tmp_path) -> TestClient:
    isolated = AdvancedRagService(
        storage_file=str(tmp_path / "corpus.json"),
        embedding_dimensions=128,
        chunk_size=80,
        chunk_overlap=10,
        bm25_k1=1.5,
        bm25_b=0.75,
        rrf_k=60,
        reranker_enabled=True,
        answer_backend="rule_based",
    )
    previous = dependencies._rag_service
    dependencies._rag_service = isolated
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        dependencies._rag_service = previous
