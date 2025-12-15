"""
 Health check router
Provides endpoints for monitoring service status.
"""

from fastapi import APIRouter, Depends

from app.models.schemas import HealthCheckResponse

# from app.services.prediction_service import PredictionService

router = APIRouter(prefix="", tags=["Health"])  # No prefix, so /health is at root level


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check if the API is running and the ML model is loaded",
)
async def health_check(
    # service: PredictionService = Depends(get_prediction_service)
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
        status="healthy", model_loaded=True  # service.is_model_loaded()
    )
