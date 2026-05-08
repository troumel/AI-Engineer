"""Persistent storage for uploaded images and their metadata."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Iterable


_EXTENSION_MAP = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
}


class ImageRecord:
    """In-memory representation of an image's metadata."""

    def __init__(
        self,
        image_id: str,
        filename: str,
        content_type: str,
        size_bytes: int,
        storage_path: str,
        created_at: str,
        tags: list[str],
        caption: str | None = None,
    ) -> None:
        self.image_id = image_id
        self.filename = filename
        self.content_type = content_type
        self.size_bytes = size_bytes
        self.storage_path = storage_path
        self.created_at = created_at
        self.tags = list(tags)
        self.caption = caption

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "filename": self.filename,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "storage_path": self.storage_path,
            "created_at": self.created_at,
            "tags": self.tags,
            "caption": self.caption,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ImageRecord":
        return cls(
            image_id=payload["image_id"],
            filename=payload["filename"],
            content_type=payload["content_type"],
            size_bytes=int(payload["size_bytes"]),
            storage_path=payload["storage_path"],
            created_at=payload["created_at"],
            tags=list(payload.get("tags", [])),
            caption=payload.get("caption"),
        )


class ImageStore:
    """Persist uploaded images on disk and maintain a JSON metadata index."""

    def __init__(self, images_directory: str, metadata_file: str) -> None:
        self.images_directory = Path(images_directory)
        self.metadata_path = Path(metadata_file)
        self.images_directory.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._records: dict[str, ImageRecord] = {}
        self._load()

    def save_image(
        self,
        filename: str,
        content_type: str,
        data: bytes,
        tags: Iterable[str],
    ) -> ImageRecord:
        """Persist image bytes to disk and append metadata."""
        image_id = uuid.uuid4().hex
        extension = self._derive_extension(filename, content_type)
        storage_path = self.images_directory / f"{image_id}{extension}"
        storage_path.write_bytes(data)

        record = ImageRecord(
            image_id=image_id,
            filename=filename or storage_path.name,
            content_type=content_type or "application/octet-stream",
            size_bytes=len(data),
            storage_path=str(storage_path),
            created_at=datetime.now(timezone.utc).isoformat(),
            tags=self._normalize_tags(tags),
        )
        with self._lock:
            self._records[image_id] = record
            self._persist_unlocked()
        return record

    def get(self, image_id: str) -> ImageRecord:
        record = self._records.get(image_id)
        if record is None:
            raise KeyError(f"Image '{image_id}' was not found.")
        return record

    def read_bytes(self, image_id: str) -> bytes:
        record = self.get(image_id)
        return Path(record.storage_path).read_bytes()

    def list_records(self) -> list[ImageRecord]:
        return sorted(self._records.values(), key=lambda r: r.created_at)

    def update_caption(self, image_id: str, caption: str) -> ImageRecord:
        with self._lock:
            record = self._records.get(image_id)
            if record is None:
                raise KeyError(f"Image '{image_id}' was not found.")
            record.caption = caption
            self._persist_unlocked()
            return record

    def __len__(self) -> int:
        return len(self._records)

    def _normalize_tags(self, tags: Iterable[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            cleaned = (tag or "").strip().lower()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                normalized.append(cleaned)
        return normalized

    def _derive_extension(self, filename: str, content_type: str) -> str:
        if filename:
            suffix = Path(filename).suffix.lower()
            if re.fullmatch(r"\.[a-z0-9]{1,5}", suffix or ""):
                return suffix
        return _EXTENSION_MAP.get((content_type or "").lower(), ".bin")

    def _load(self) -> None:
        if not self.metadata_path.exists():
            return
        try:
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        for entry in payload.get("images", []):
            try:
                record = ImageRecord.from_dict(entry)
            except (KeyError, ValueError):
                continue
            self._records[record.image_id] = record

    def _persist_unlocked(self) -> None:
        payload = {"images": [record.to_dict() for record in self._records.values()]}
        self.metadata_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
