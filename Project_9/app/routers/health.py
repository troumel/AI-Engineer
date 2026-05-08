"""Health check router."""

from fastapi import APIRouter, Depends

from app.dependencies import get_inference_service
from app.models.schemas import HealthCheckResponse
from app.services.inference_service import InferenceService

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    service: InferenceService = Depends(get_inference_service),
) -> HealthCheckResponse:
    """Return service health and artifact status."""
    return service.get_health()
