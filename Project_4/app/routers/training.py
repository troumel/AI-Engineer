"""Training job endpoints for Project 4."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_ner_service
from app.models.schemas import (
    TrainingJobRequest,
    TrainingJobResponse,
    TrainingJobStatusResponse,
)
from app.services.ner_service import NERService


router = APIRouter(prefix="/training", tags=["Training"])


@router.post(
    "/jobs",
    response_model=TrainingJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create Training Job",
    description="Trigger a model training job from annotated examples.",
)
async def create_training_job(
    request: TrainingJobRequest,
    service: NERService = Depends(get_ner_service),
) -> TrainingJobResponse:
    """Create a training job for a new model version."""
    try:
        return service.start_training_job(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get(
    "/jobs/{job_id}",
    response_model=TrainingJobStatusResponse,
    summary="Get Training Job Status",
    description="Return status and progress for a training job.",
)
async def get_training_job_status(
    job_id: str,
    service: NERService = Depends(get_ner_service),
) -> TrainingJobStatusResponse:
    """Get training job status by id."""
    try:
        return service.get_job_status(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))