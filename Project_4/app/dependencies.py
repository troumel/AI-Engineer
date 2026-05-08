"""Dependency registration and singleton initialization for Project 4."""

from typing import Optional

from app.config import settings
from app.services.ner_service import NERService


_ner_service: Optional[NERService] = None


def initialize_services() -> None:
    """Initialize the singleton NER service once at startup."""
    global _ner_service

    if _ner_service is not None:
        return

    _ner_service = NERService(
        models_directory=settings.models_directory,
        default_model_version=settings.default_model_version,
        base_model_name=settings.base_model_name,
        max_training_workers=settings.max_training_workers,
        default_candidate_percentage=settings.default_candidate_percentage,
    )


def get_ner_service() -> NERService:
    """Return the singleton NER service instance."""
    if _ner_service is None:
        raise RuntimeError("NERService not initialized. Call initialize_services() at app startup.")

    return _ner_service