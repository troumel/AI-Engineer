"""Health and root routes."""

from fastapi import APIRouter, Depends

from app.dependencies import get_llm_service
from app.models.schemas import HealthCheckResponse
from app.services.llm_service import LLMService

router = APIRouter()


@router.get("/", tags=["health"])
def root() -> dict[str, str]:
    return {
        "service": "custom-llm-inference-api",
        "openai_compatible": "/v1",
        "metrics": "/metrics",
        "docs": "/docs",
    }


@router.get("/health", response_model=HealthCheckResponse, tags=["health"])
def health(
    llm_service: LLMService = Depends(get_llm_service),
) -> HealthCheckResponse:
    return llm_service.get_health()
