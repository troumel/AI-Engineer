"""Persistent corpus storage with chunk-level metadata."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Iterable


@dataclass
class ChunkRecord:
    """One indexed chunk."""

    chunk_id: str
    document_id: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentRecord:
    """A logical document made of chunk records."""

    document_id: str
    title: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_ids: list[str] = field(default_factory=list)


class CorpusStore:
    """JSON-backed corpus persisting documents + chunks."""

    def __init__(self, storage_path: str) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._documents: dict[str, DocumentRecord] = {}
        self._chunks: dict[str, ChunkRecord] = {}
        self._load()

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add_document(
        self,
        title: str,
        chunks: Iterable[str],
        metadata: dict[str, Any],
        document_id: str | None = None,
    ) -> tuple[DocumentRecord, list[ChunkRecord]]:
        document_id = document_id or uuid.uuid4().hex
        chunk_records: list[ChunkRecord] = []

        with self._lock:
            if document_id in self._documents:
                raise ValueError(f"Document '{document_id}' already exists.")

            for index, content in enumerate(chunks):
                chunk_id = f"{document_id}::{index}"
                record = ChunkRecord(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    title=title,
                    content=content,
                    metadata=dict(metadata),
                )
                chunk_records.append(record)
                self._chunks[chunk_id] = record

            document = DocumentRecord(
                document_id=document_id,
                title=title,
                created_at=datetime.now(timezone.utc).isoformat(),
                metadata=dict(metadata),
                chunk_ids=[record.chunk_id for record in chunk_records],
            )
            self._documents[document_id] = document
            self._persist_unlocked()

        return document, chunk_records

    def remove_document(self, document_id: str) -> list[str]:
        with self._lock:
            document = self._documents.pop(document_id, None)
            if document is None:
                raise KeyError(f"Document '{document_id}' was not found.")
            removed_chunk_ids = list(document.chunk_ids)
            for chunk_id in removed_chunk_ids:
                self._chunks.pop(chunk_id, None)
            self._persist_unlocked()
        return removed_chunk_ids

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def list_documents(self) -> list[DocumentRecord]:
        return sorted(
            self._documents.values(), key=lambda doc: doc.created_at, reverse=True
        )

    def list_chunks(self) -> list[ChunkRecord]:
        return list(self._chunks.values())

    def get_chunk(self, chunk_id: str) -> ChunkRecord:
        record = self._chunks.get(chunk_id)
        if record is None:
            raise KeyError(f"Chunk '{chunk_id}' was not found.")
        return record

    def chunk_count(self) -> int:
        return len(self._chunks)

    def document_count(self) -> int:
        return len(self._documents)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return

        for entry in payload.get("documents", []):
            try:
                document = DocumentRecord(
                    document_id=entry["document_id"],
                    title=entry["title"],
                    created_at=entry["created_at"],
                    metadata=dict(entry.get("metadata", {})),
                    chunk_ids=list(entry.get("chunk_ids", [])),
                )
            except KeyError:
                continue
            self._documents[document.document_id] = document

        for entry in payload.get("chunks", []):
            try:
                chunk = ChunkRecord(
                    chunk_id=entry["chunk_id"],
                    document_id=entry["document_id"],
                    title=entry["title"],
                    content=entry["content"],
                    metadata=dict(entry.get("metadata", {})),
                )
            except KeyError:
                continue
            self._chunks[chunk.chunk_id] = chunk

    def _persist_unlocked(self) -> None:
        payload = {
            "documents": [asdict(document) for document in self._documents.values()],
            "chunks": [asdict(chunk) for chunk in self._chunks.values()],
        }
        self.storage_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
