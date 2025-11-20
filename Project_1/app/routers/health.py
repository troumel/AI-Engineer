"""
Health check router.

Provides endpoints for monitoring service status.
Similar to health check endpoints in .NET (IHealthCheck).
"""

from fastapi import APIRouter, Depends

from app.models.schemas import HealthCheckResponse
from app.services.prediction_service import PredictionService
from app.dependencies import get_prediction_service


# Create a router (like a Controller in .NET)
# prefix="/health" means all routes here start with /health
# tags=["Health"] groups these endpoints in the API docs
router = APIRouter(
    prefix="",  # No prefix, so /health is at root level
    tags=["Health"]
)


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check if the API is running and the ML model is loaded"
)
async def health_check(
    service: PredictionService = Depends(get_prediction_service)
) -> HealthCheckResponse:
    """
    Health check endpoint.

    Returns the service status and whether the ML model is loaded.
    Useful for:
    - Load balancer health checks
    - Kubernetes liveness/readiness probes
    - Monitoring systems

    Args:
        service: PredictionService injected via dependency injection

    Returns:
        HealthCheckResponse with status and model_loaded flag
    """
    return HealthCheckResponse(
        status="healthy",
        model_loaded=service.is_model_loaded()
    )
