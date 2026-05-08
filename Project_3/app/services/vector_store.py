"""Vector store implementations for retrieved document chunks."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DocumentChunkRecord:
    """Chunk plus embedding ready to be stored."""

    chunk_id: str
    document_id: str
    source_name: str
    chunk_index: int
    content: str
    embedding: list[float]


@dataclass(frozen=True)
class RetrievedChunk:
    """Retrieved chunk with similarity score."""

    chunk_id: str
    document_id: str
    source_name: str
    chunk_index: int
    content: str
    score: float


class VectorStore(ABC):
    """Abstract vector store used by the RAG service."""

    backend_name: str = "unknown"

    @abstractmethod
    def upsert_chunks(self, records: list[DocumentChunkRecord]) -> None:
        """Store chunks and embeddings."""

    @abstractmethod
    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """Return the most similar chunks for the query embedding."""

    @abstractmethod
    def count_chunks(self) -> int:
        """Return the total number of stored chunks."""


class InMemoryVectorStore(VectorStore):
    """Simple in-memory cosine-similarity vector store."""

    backend_name = "memory"

    def __init__(self):
        self._records: list[DocumentChunkRecord] = []

    def upsert_chunks(self, records: list[DocumentChunkRecord]) -> None:
        record_ids = {record.chunk_id for record in records}
        self._records = [record for record in self._records if record.chunk_id not in record_ids]
        self._records.extend(records)

    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        if not self._records:
            return []

        query_vector = np.array(query_embedding, dtype=float)
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return []

        scored_records: list[RetrievedChunk] = []
        for record in self._records:
            embedding = np.array(record.embedding, dtype=float)
            denominator = np.linalg.norm(embedding) * query_norm
            similarity = float(np.dot(query_vector, embedding) / denominator) if denominator else 0.0
            similarity = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
            scored_records.append(
                RetrievedChunk(
                    chunk_id=record.chunk_id,
                    document_id=record.document_id,
                    source_name=record.source_name,
                    chunk_index=record.chunk_index,
                    content=record.content,
                    score=similarity,
                )
            )

        scored_records.sort(key=lambda item: item.score, reverse=True)
        return scored_records[:top_k]

    def count_chunks(self) -> int:
        return len(self._records)


class ChromaVectorStore(VectorStore):
    """Persistent ChromaDB-backed vector store."""

    backend_name = "chroma"

    def __init__(self, persist_directory: str, collection_name: str):
        chromadb_module = importlib.import_module("chromadb")
        client_type = getattr(chromadb_module, "PersistentClient")
        self.client = client_type(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def upsert_chunks(self, records: list[DocumentChunkRecord]) -> None:
        if not records:
            return

        self.collection.upsert(
            ids=[record.chunk_id for record in records],
            documents=[record.content for record in records],
            embeddings=[record.embedding for record in records],
            metadatas=[
                {
                    "document_id": record.document_id,
                    "source_name": record.source_name,
                    "chunk_index": record.chunk_index,
                }
                for record in records
            ],
        )

    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        response = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        ids = response.get("ids", [[]])[0]

        results: list[RetrievedChunk] = []
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances, strict=False):
            similarity = 1.0 / (1.0 + float(distance))
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=str(metadata["document_id"]),
                    source_name=str(metadata["source_name"]),
                    chunk_index=int(metadata["chunk_index"]),
                    content=str(document),
                    score=max(0.0, min(1.0, similarity)),
                )
            )
        return results

    def count_chunks(self) -> int:
        return int(self.collection.count())


def create_vector_store(
    preferred_backend: str,
    persist_directory: str,
    collection_name: str,
) -> VectorStore:
    """Create the configured vector store, falling back to memory when needed."""
    if preferred_backend.lower() == "chroma":
        try:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            return ChromaVectorStore(
                persist_directory=persist_directory,
                collection_name=collection_name,
            )
        except Exception:
            return InMemoryVectorStore()

    return InMemoryVectorStore()