"""Health check router."""

from fastapi import APIRouter, Depends

from app.dependencies import get_multimodal_service
from app.models.schemas import HealthCheckResponse
from app.services.multimodal_service import MultiModalService


router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    service: MultiModalService = Depends(get_multimodal_service),
) -> HealthCheckResponse:
    """Return service health and configured vision provider."""
    return service.get_health()
