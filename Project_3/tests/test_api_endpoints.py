"""Integration tests for the Project 3 API endpoints."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import dependencies
from app.main import app
from app.services.answer_generator import AnswerGenerator
from app.services.document_processor import DocumentProcessor
from app.services.embedding_provider import EmbeddingProvider
from app.services.rag_service import RagService
from app.services.vector_store import InMemoryVectorStore, RetrievedChunk


class FakeEmbeddingProvider(EmbeddingProvider):
    """Simple keyword-based embedding provider for deterministic tests."""

    provider_name = "fake-embeddings"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "python" in lowered else 0.0,
                    1.0 if "fastapi" in lowered else 0.0,
                    1.0 if "rag" in lowered else 0.0,
                    1.0 if "vector" in lowered else 0.0,
                ]
            )
        return vectors


class FakeAnswerGenerator(AnswerGenerator):
    """Deterministic answer generator for endpoint tests."""

    provider_name = "fake-answer-generator"

    def generate_answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "No answer found."
        return f"Answer based on {chunks[0].source_name}: {chunks[0].content[:120]}"


@pytest.fixture(autouse=True)
def initialize_test_services(tmp_path: Path):
    """Inject an in-memory RAG service before each test."""
    processor = DocumentProcessor(
        upload_directory=str(tmp_path / "uploads"),
        chunk_size=160,
        chunk_overlap=20,
    )
    dependencies._rag_service = RagService(
        document_processor=processor,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=InMemoryVectorStore(),
        answer_generator=FakeAnswerGenerator(),
        default_top_k=3,
    )
    yield
    dependencies._rag_service = None


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_returns_api_info(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_check_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["indexed_documents"] == 0


def test_upload_text_document_returns_201(client):
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "intro.txt",
                b"Python and FastAPI are useful for building a RAG system with vector search.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["filename"] == "intro.txt"
    assert payload["chunks_indexed"] >= 1


def test_upload_rejects_unsupported_file_type(client):
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.docx", b"not supported", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )

    assert response.status_code == 400


def test_list_documents_returns_uploaded_document(client):
    client.post(
        "/documents/upload",
        files={"file": ("guide.txt", b"RAG systems combine retrieval with generation.", "text/plain")},
    )

    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json()["total_documents"] == 1


def test_qa_endpoint_returns_answer_with_citations(client):
    client.post(
        "/documents/upload",
        files={
            "file": (
                "guide.txt",
                b"Python and FastAPI are often used to build a RAG system. Vector search retrieves relevant chunks.",
                "text/plain",
            )
        },
    )

    response = client.post("/qa/ask", json={"question": "How is FastAPI used in a RAG system?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval_count"] >= 1
    assert payload["answer_provider"] == "fake-answer-generator"
    assert payload["citations"][0]["source_name"] == "guide.txt"


def test_qa_endpoint_requires_uploaded_documents(client):
    response = client.post("/qa/ask", json={"question": "What is RAG?"})

    assert response.status_code == 400