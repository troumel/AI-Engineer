"""Dependency registration and singleton initialization for Project 9."""

from typing import Optional

from app.config import settings
from app.services.inference_service import InferenceService

_inference_service: Optional[InferenceService] = None


def initialize_services() -> None:
    """Initialize shared services once at application startup."""
    global _inference_service

    if _inference_service is not None:
        return

    _inference_service = InferenceService(
        models_directory=settings.models_directory,
        default_model_version=settings.default_model_version,
        base_model_name=settings.base_model_name,
    )
    _inference_service.load_model(
        allow_degraded_startup=settings.allow_degraded_startup,
    )


def get_inference_service() -> InferenceService:
    """Return the singleton inference service instance."""
    if _inference_service is None:
        raise RuntimeError(
            "InferenceService not initialized. Call initialize_services() at app startup."
        )

    return _inference_service
