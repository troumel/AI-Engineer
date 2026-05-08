"""Health and root routes."""

from fastapi import APIRouter, Depends

from app.dependencies import get_rag_service
from app.models.schemas import HealthCheckResponse
from app.services.rag_service import AdvancedRagService

router = APIRouter()


@router.get("/", tags=["health"])
def root() -> dict[str, str]:
    return {
        "service": "advanced-rag-api",
        "docs": "/docs",
        "health": "/health",
    }


@router.get("/health", response_model=HealthCheckResponse, tags=["health"])
def health(
    rag_service: AdvancedRagService = Depends(get_rag_service),
) -> HealthCheckResponse:
    return rag_service.get_health()
