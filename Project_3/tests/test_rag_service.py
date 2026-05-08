"""Unit tests for the Project 3 RAG service."""

from pathlib import Path

from app.services.answer_generator import AnswerGenerator
from app.services.document_processor import DocumentProcessor
from app.services.embedding_provider import EmbeddingProvider
from app.services.rag_service import RagService
from app.services.vector_store import InMemoryVectorStore, RetrievedChunk


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider for service tests."""

    provider_name = "fake-embeddings"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    2.0 if "python" in lowered else 0.0,
                    2.0 if "fastapi" in lowered else 0.0,
                    2.0 if "rag" in lowered else 0.0,
                    2.0 if "retrieval" in lowered else 0.0,
                ]
            )
        return vectors


class FakeAnswerGenerator(AnswerGenerator):
    """Deterministic answer generator for service tests."""

    provider_name = "fake-answer-generator"

    def generate_answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "No context found."
        return f"Using {chunks[0].source_name}, answer the question: {question}"


def build_service(tmp_path: Path) -> RagService:
    processor = DocumentProcessor(
        upload_directory=str(tmp_path / "uploads"),
        chunk_size=120,
        chunk_overlap=20,
    )
    return RagService(
        document_processor=processor,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=InMemoryVectorStore(),
        answer_generator=FakeAnswerGenerator(),
        default_top_k=3,
    )


def test_ingest_document_indexes_chunks_and_persists_manifest(tmp_path: Path):
    service = build_service(tmp_path)

    response = service.ingest_document(
        filename="guide.txt",
        content_type="text/plain",
        data=b"Python and FastAPI can power a RAG system with retrieval and generation.",
    )

    assert response.filename == "guide.txt"
    assert response.chunks_indexed >= 1
    assert (tmp_path / "uploads" / "documents.json").exists()


def test_answer_question_returns_citations(tmp_path: Path):
    service = build_service(tmp_path)
    service.ingest_document(
        filename="guide.txt",
        content_type="text/plain",
        data=b"Python and FastAPI can power a RAG system with retrieval and generation.",
    )

    response = service.answer_question("How does FastAPI fit into RAG?", top_k=2)

    assert response.retrieval_count >= 1
    assert response.citations[0].source_name == "guide.txt"


def test_answer_question_raises_without_documents(tmp_path: Path):
    service = build_service(tmp_path)

    try:
        service.answer_question("What is RAG?", top_k=2)
        raised = False
    except ValueError:
        raised = True

    assert raised is True


def test_list_documents_returns_indexed_metadata(tmp_path: Path):
    service = build_service(tmp_path)
    service.ingest_document(
        filename="guide.txt",
        content_type="text/plain",
        data=b"RAG combines retrieval and generation.",
    )

    response = service.list_documents()

    assert response.total_documents == 1
    assert response.documents[0].filename == "guide.txt"