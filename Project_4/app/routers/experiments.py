"""A/B rollout endpoints for Project 4."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_ner_service
from app.models.schemas import RolloutConfigRequest, RolloutConfigResponse
from app.services.ner_service import NERService


router = APIRouter(prefix="/experiments", tags=["Experiments"])


@router.post(
    "/rollout",
    response_model=RolloutConfigResponse,
    summary="Configure A/B Rollout",
    description="Configure primary and candidate model versions for deterministic A/B routing.",
)
async def configure_rollout(
    request: RolloutConfigRequest,
    service: NERService = Depends(get_ner_service),
) -> RolloutConfigResponse:
    """Update the rollout configuration."""
    try:
        return service.configure_rollout(
            primary_version=request.primary_version,
            candidate_version=request.candidate_version,
            candidate_percentage=request.candidate_percentage,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))