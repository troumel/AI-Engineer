"""Health check endpoints for the RAG API."""

from fastapi import APIRouter, Depends

from app.dependencies import get_rag_service
from app.models.schemas import HealthCheckResponse
from app.services.rag_service import RagService


router = APIRouter(prefix="", tags=["Health"])


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check indexing status and which providers are active.",
)
async def health_check(service: RagService = Depends(get_rag_service)) -> HealthCheckResponse:
    """Return service health details."""
    return service.get_health_status()