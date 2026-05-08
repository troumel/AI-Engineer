"""Document parsing, saving, and chunking for uploaded files."""

from __future__ import annotations

import importlib
import io
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class ProcessedDocument:
    """Normalized document produced from an upload."""

    document_id: str
    filename: str
    content_type: str
    stored_path: str
    text: str
    chunks: list[str]


class DocumentProcessor:
    """Parse PDF/TXT uploads and split them into retrievable chunks."""

    def __init__(self, upload_directory: str, chunk_size: int, chunk_overlap: int):
        self.upload_directory = Path(upload_directory)
        self.upload_directory.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_upload(self, filename: str, content_type: str | None, data: bytes) -> ProcessedDocument:
        """Validate, save, parse, and chunk an uploaded document."""
        suffix = Path(filename).suffix.lower()
        if suffix not in {".txt", ".pdf"}:
            raise ValueError("Only .txt and .pdf files are supported.")

        document_id = uuid4().hex
        stored_filename = f"{document_id}_{Path(filename).name}"
        stored_path = self.upload_directory / stored_filename
        stored_path.write_bytes(data)

        detected_type = content_type or self._detect_content_type(suffix)
        if suffix == ".txt":
            text = data.decode("utf-8")
        else:
            text = self._extract_pdf_text(data)

        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("The uploaded document did not contain readable text.")

        chunks = self._chunk_text(normalized_text)
        if not chunks:
            raise ValueError("The uploaded document did not produce retrievable chunks.")

        return ProcessedDocument(
            document_id=document_id,
            filename=Path(filename).name,
            content_type=detected_type,
            stored_path=str(stored_path),
            text=normalized_text,
            chunks=chunks,
        )

    def _detect_content_type(self, suffix: str) -> str:
        return "application/pdf" if suffix == ".pdf" else "text/plain"

    def _extract_pdf_text(self, data: bytes) -> str:
        pypdf2 = importlib.import_module("PyPDF2")
        reader_type = getattr(pypdf2, "PdfReader")
        reader = reader_type(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    def _chunk_text(self, text: str) -> list[str]:
        try:
            splitter_module = importlib.import_module("langchain_text_splitters")
            splitter_type = getattr(splitter_module, "RecursiveCharacterTextSplitter")
            splitter = splitter_type(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            chunks = splitter.split_text(text)
            return [chunk.strip() for chunk in chunks if chunk.strip()]
        except Exception:
            return self._manual_chunk_text(text)

    def _manual_chunk_text(self, text: str) -> list[str]:
        chunks: list[str] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for start in range(0, len(text), step):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
        return chunks