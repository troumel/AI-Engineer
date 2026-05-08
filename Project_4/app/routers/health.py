"""Health check endpoints for the NER API."""

from fastapi import APIRouter, Depends

from app.dependencies import get_ner_service
from app.models.schemas import HealthCheckResponse
from app.services.ner_service import NERService


router = APIRouter(prefix="", tags=["Health"])


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check current model registry and training job status.",
)
async def health_check(service: NERService = Depends(get_ner_service)) -> HealthCheckResponse:
    """Return health information for the service."""
    return service.get_health_status()