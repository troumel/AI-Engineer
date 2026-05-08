"""High-level RAG orchestration service."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from app.models.schemas import (
    AnswerResponse,
    DocumentSummary,
    DocumentsResponse,
    DocumentUploadResponse,
    HealthCheckResponse,
    SourceCitation,
)
from app.services.answer_generator import AnswerGenerator
from app.services.document_processor import DocumentProcessor
from app.services.embedding_provider import EmbeddingProvider
from app.services.vector_store import DocumentChunkRecord, RetrievedChunk, VectorStore


@dataclass
class DocumentManifestEntry:
    """Metadata persisted for each indexed document."""

    document_id: str
    filename: str
    content_type: str
    chunks_indexed: int
    stored_path: str
    uploaded_at: str


class RagService:
    """Coordinate ingestion, retrieval, and grounded answer generation."""

    def __init__(
        self,
        document_processor: DocumentProcessor,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        answer_generator: AnswerGenerator,
        default_top_k: int,
    ):
        self.document_processor = document_processor
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.answer_generator = answer_generator
        self.default_top_k = default_top_k
        self.manifest_path = Path(self.document_processor.upload_directory) / "documents.json"
        self._lock = Lock()
        self._documents = self._load_manifest()

    def ingest_document(
        self,
        filename: str,
        content_type: str | None,
        data: bytes,
    ) -> DocumentUploadResponse:
        """Ingest one uploaded document into the vector store."""
        processed = self.document_processor.process_upload(
            filename=filename,
            content_type=content_type,
            data=data,
        )
        embeddings = self.embedding_provider.embed_texts(processed.chunks)
        records = [
            DocumentChunkRecord(
                chunk_id=f"{processed.document_id}:{index}",
                document_id=processed.document_id,
                source_name=processed.filename,
                chunk_index=index,
                content=chunk,
                embedding=embedding,
            )
            for index, (chunk, embedding) in enumerate(zip(processed.chunks, embeddings, strict=True))
        ]
        self.vector_store.upsert_chunks(records)

        manifest_entry = DocumentManifestEntry(
            document_id=processed.document_id,
            filename=processed.filename,
            content_type=processed.content_type,
            chunks_indexed=len(processed.chunks),
            stored_path=processed.stored_path,
            uploaded_at=datetime.now(timezone.utc).isoformat(),
        )
        self._store_document(manifest_entry)

        return DocumentUploadResponse(
            document_id=processed.document_id,
            filename=processed.filename,
            content_type=processed.content_type,
            chunks_indexed=len(processed.chunks),
            stored_path=processed.stored_path,
        )

    def list_documents(self) -> DocumentsResponse:
        """List indexed documents from the manifest."""
        documents = [
            DocumentSummary(
                document_id=entry.document_id,
                filename=entry.filename,
                content_type=entry.content_type,
                chunks_indexed=entry.chunks_indexed,
                uploaded_at=entry.uploaded_at,
            )
            for entry in self._documents.values()
        ]
        return DocumentsResponse(total_documents=len(documents), documents=documents)

    def answer_question(self, question: str, top_k: int | None) -> AnswerResponse:
        """Retrieve relevant chunks and generate a grounded answer."""
        if not self._documents:
            raise ValueError("No documents have been indexed yet. Upload a document first.")

        effective_top_k = top_k or self.default_top_k
        query_embedding = self.embedding_provider.embed_texts([question])[0]
        retrieved = self.vector_store.query(query_embedding=query_embedding, top_k=effective_top_k)
        answer = self.answer_generator.generate_answer(question=question, chunks=retrieved)
        citations = [self._build_citation(chunk) for chunk in retrieved]

        return AnswerResponse(
            question=question,
            answer=answer,
            retrieval_count=len(retrieved),
            answer_provider=self.answer_generator.provider_name,
            embedding_provider=self.embedding_provider.provider_name,
            citations=citations,
        )

    def get_health_status(self) -> HealthCheckResponse:
        """Return current provider and indexing status."""
        document_count = len(self._documents)
        chunk_count = self.vector_store.count_chunks()
        return HealthCheckResponse(
            status="healthy",
            vector_store_backend=self.vector_store.backend_name,
            embedding_provider=self.embedding_provider.provider_name,
            answer_provider=self.answer_generator.provider_name,
            indexed_documents=document_count,
            indexed_chunks=chunk_count,
        )

    def _build_citation(self, chunk: RetrievedChunk) -> SourceCitation:
        return SourceCitation(
            document_id=chunk.document_id,
            source_name=chunk.source_name,
            chunk_index=chunk.chunk_index,
            similarity_score=round(chunk.score, 4),
            excerpt=chunk.content[:240],
        )

    def _load_manifest(self) -> dict[str, DocumentManifestEntry]:
        if not self.manifest_path.exists():
            return {}

        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {
            entry["document_id"]: DocumentManifestEntry(**entry)
            for entry in payload
        }

    def _store_document(self, entry: DocumentManifestEntry) -> None:
        with self._lock:
            self._documents[entry.document_id] = entry
            serialized = [asdict(item) for item in self._documents.values()]
            self.manifest_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")