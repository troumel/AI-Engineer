"""Dependency registration and singleton initialization for Project 7."""

from typing import Optional

from app.config import settings
from app.services.rag_service import AdvancedRagService


_rag_service: Optional[AdvancedRagService] = None


def initialize_services() -> None:
    global _rag_service

    if _rag_service is not None:
        return

    _rag_service = AdvancedRagService(
        storage_file=settings.storage_file,
        embedding_dimensions=settings.embedding_dimensions,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        bm25_k1=settings.bm25_k1,
        bm25_b=settings.bm25_b,
        rrf_k=settings.rrf_k,
        reranker_enabled=settings.reranker_enabled,
        answer_backend=settings.answer_backend,
    )


def get_rag_service() -> AdvancedRagService:
    if _rag_service is None:
        raise RuntimeError(
            "AdvancedRagService not initialized. Call initialize_services() at app startup."
        )
    return _rag_service
