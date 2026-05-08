"""Dependency registration and singleton initialization for Project 3."""

from typing import Optional

from app.config import settings
from app.services.answer_generator import create_answer_generator
from app.services.document_processor import DocumentProcessor
from app.services.embedding_provider import create_embedding_provider
from app.services.rag_service import RagService
from app.services.vector_store import create_vector_store


_rag_service: Optional[RagService] = None


def initialize_services() -> None:
    """Initialize the singleton RAG service once at application startup."""
    global _rag_service

    if _rag_service is not None:
        return

    processor = DocumentProcessor(
        upload_directory=settings.upload_directory,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embedding_provider = create_embedding_provider(
        api_key=settings.openai_api_key,
        model_name=settings.openai_embedding_model,
    )
    vector_store = create_vector_store(
        preferred_backend=settings.vector_store_backend,
        persist_directory=settings.chroma_persist_directory,
        collection_name=settings.chroma_collection_name,
    )
    answer_generator = create_answer_generator(
        api_key=settings.openai_api_key,
        model_name=settings.openai_chat_model,
    )

    _rag_service = RagService(
        document_processor=processor,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        answer_generator=answer_generator,
        default_top_k=settings.retrieval_top_k,
    )


def get_rag_service() -> RagService:
    """Return the singleton RAG service instance."""
    if _rag_service is None:
        raise RuntimeError("RagService not initialized. Call initialize_services() at app startup.")

    return _rag_service