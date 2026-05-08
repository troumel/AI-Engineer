"""Health check endpoints for the sentiment analysis API."""

from fastapi import APIRouter, Depends

from app.dependencies import get_sentiment_service
from app.models.schemas import HealthCheckResponse
from app.services.sentiment_service import SentimentService


router = APIRouter(prefix="", tags=["Health"])


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check whether the API is running and the classifier is available.",
)
async def health_check(
    service: SentimentService = Depends(get_sentiment_service),
) -> HealthCheckResponse:
    """Return service, model, and cache status."""
    return HealthCheckResponse(
        status="healthy" if service.is_model_loaded() else "degraded",
        model_loaded=service.is_model_loaded(),
        cache_backend=service.cache_backend_name,
        model_name=service.model_name,
    )