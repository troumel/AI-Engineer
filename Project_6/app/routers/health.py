"""Health check router."""

from fastapi import APIRouter, Depends

from app.dependencies import get_agent_service
from app.models.schemas import HealthCheckResponse
from app.services.agent_service import AgentService


router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    service: AgentService = Depends(get_agent_service),
) -> HealthCheckResponse:
    """Return service health and configured planner backend."""
    return service.get_health()
