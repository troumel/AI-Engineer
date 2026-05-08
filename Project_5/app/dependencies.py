"""Dependency registration and singleton initialization for Project 5."""

from typing import Optional

from app.config import settings
from app.services.multimodal_service import MultiModalService


_multimodal_service: Optional[MultiModalService] = None


def initialize_services() -> None:
    """Initialize the singleton multi-modal service once at startup."""
    global _multimodal_service

    if _multimodal_service is not None:
        return

    _multimodal_service = MultiModalService(
        images_directory=settings.images_directory,
        metadata_file=settings.metadata_file,
        vision_provider=settings.vision_provider,
        embedding_dimensions=settings.embedding_dimensions,
        max_upload_bytes=settings.max_upload_bytes,
    )


def get_multimodal_service() -> MultiModalService:
    """Return the singleton multi-modal service instance."""
    if _multimodal_service is None:
        raise RuntimeError(
            "MultiModalService not initialized. Call initialize_services() at app startup."
        )
    return _multimodal_service
