"""High-level multi-modal orchestration service."""

from __future__ import annotations

from pathlib import Path

from app.models.schemas import (
    CaptionResponse,
    HealthCheckResponse,
    ImageInfo,
    ImageListResponse,
    ImageUploadResponse,
    SearchResponse,
    SearchResultItem,
    VQAResponse,
)
from app.services.image_store import ImageRecord, ImageStore
from app.services.vector_index import VectorIndex
from app.services.vision_provider import VisionProvider, build_vision_provider


class MultiModalService:
    """Coordinate the image store, vector index, and vision provider."""

    def __init__(
        self,
        images_directory: str,
        metadata_file: str,
        vision_provider: str,
        embedding_dimensions: int,
        max_upload_bytes: int,
    ) -> None:
        self.max_upload_bytes = max_upload_bytes
        self.embedding_dimensions = embedding_dimensions

        self.image_store = ImageStore(
            images_directory=images_directory,
            metadata_file=metadata_file,
        )
        self.vision: VisionProvider = build_vision_provider(
            name=vision_provider,
            dimensions=embedding_dimensions,
        )
        self.index = VectorIndex()
        self._reindex_existing_records()

    def upload_image(
        self,
        filename: str,
        content_type: str,
        data: bytes,
        tags: list[str],
    ) -> ImageUploadResponse:
        """Persist an image and add it to the multi-modal index."""
        if not data:
            raise ValueError("Uploaded file is empty.")
        if len(data) > self.max_upload_bytes:
            raise ValueError(
                f"Uploaded file exceeds maximum size of {self.max_upload_bytes} bytes."
            )
        if content_type and not content_type.lower().startswith("image/"):
            raise ValueError(
                f"Unsupported content type '{content_type}'. Expected an image/* type."
            )

        record = self.image_store.save_image(
            filename=filename,
            content_type=content_type,
            data=data,
            tags=tags,
        )
        embedding = self.vision.embed_image(data, hints=self._record_hints(record))
        self.index.upsert(record.image_id, embedding)
        return ImageUploadResponse(**record.to_dict())

    def get_image(self, image_id: str) -> ImageInfo:
        record = self._require_record(image_id)
        return ImageInfo(**self._record_info_payload(record))

    def list_images(self) -> ImageListResponse:
        records = self.image_store.list_records()
        return ImageListResponse(
            count=len(records),
            images=[ImageInfo(**self._record_info_payload(record)) for record in records],
        )

    def caption_image(self, image_id: str, prompt: str | None) -> CaptionResponse:
        record = self._require_record(image_id)
        data = self._read_bytes(record)
        caption_text = self.vision.caption(
            data,
            hints=self._record_hints(record),
            prompt=prompt,
        )
        updated = self.image_store.update_caption(image_id, caption_text)
        embedding = self.vision.embed_image(
            data,
            hints=self._record_hints(updated),
        )
        self.index.upsert(updated.image_id, embedding)
        return CaptionResponse(
            image_id=updated.image_id,
            caption=caption_text,
            model=self.vision.name,
        )

    def answer_question(self, image_id: str, question: str) -> VQAResponse:
        record = self._require_record(image_id)
        data = self._read_bytes(record)
        answer_text, confidence = self.vision.answer(
            data,
            question=question,
            hints=self._record_hints(record),
        )
        return VQAResponse(
            image_id=image_id,
            question=question,
            answer=answer_text,
            confidence=round(float(confidence), 4),
            model=self.vision.name,
        )

    def search_by_text(self, query: str, top_k: int) -> SearchResponse:
        embedding = self.vision.embed_text(query)
        matches = self.index.search(embedding, top_k=top_k)
        return SearchResponse(
            query=query,
            count=len(matches),
            results=[self._build_result(image_id, score) for image_id, score in matches],
        )

    def search_by_image(self, image_id: str, top_k: int) -> SearchResponse:
        self._require_record(image_id)
        query_vector = self.index.get(image_id)
        if query_vector is None:
            raise KeyError(f"No embedding indexed for image '{image_id}'.")
        matches = self.index.search(
            query_vector,
            top_k=top_k,
            exclude_keys={image_id},
        )
        return SearchResponse(
            query=image_id,
            count=len(matches),
            results=[self._build_result(other_id, score) for other_id, score in matches],
        )

    def get_health(self) -> HealthCheckResponse:
        return HealthCheckResponse(
            status="healthy",
            vision_provider=self.vision.name,
            image_count=len(self.image_store),
            embedding_dimensions=self.embedding_dimensions,
        )

    def _record_hints(self, record: ImageRecord) -> list[str]:
        hints: list[str] = []
        hints.extend(record.tags)
        if record.caption:
            hints.append(record.caption)
        if record.filename:
            stem = Path(record.filename).stem.replace("_", " ").replace("-", " ")
            hints.append(stem)
        return hints

    def _record_info_payload(self, record: ImageRecord) -> dict:
        payload = record.to_dict()
        payload.pop("storage_path", None)
        return payload

    def _build_result(self, image_id: str, score: float) -> SearchResultItem:
        record = self.image_store.get(image_id)
        return SearchResultItem(
            image_id=record.image_id,
            filename=record.filename,
            score=round(float(score), 6),
            caption=record.caption,
            tags=record.tags,
        )

    def _require_record(self, image_id: str) -> ImageRecord:
        try:
            return self.image_store.get(image_id)
        except KeyError as exc:
            raise KeyError(str(exc)) from exc

    def _read_bytes(self, record: ImageRecord) -> bytes:
        try:
            return Path(record.storage_path).read_bytes()
        except FileNotFoundError as exc:
            raise KeyError(
                f"Stored file for image '{record.image_id}' is missing."
            ) from exc

    def _reindex_existing_records(self) -> None:
        for record in self.image_store.list_records():
            try:
                data = Path(record.storage_path).read_bytes()
            except FileNotFoundError:
                continue
            embedding = self.vision.embed_image(data, hints=self._record_hints(record))
            self.index.upsert(record.image_id, embedding)
