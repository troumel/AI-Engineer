"""Model registry endpoints for Project 4."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_ner_service
from app.models.schemas import ActivateModelRequest, ModelsResponse
from app.services.ner_service import NERService


router = APIRouter(prefix="/models", tags=["Models"])


@router.get(
    "",
    response_model=ModelsResponse,
    summary="List Model Versions",
    description="List available model versions, the active version, and rollout state.",
)
async def list_models(service: NERService = Depends(get_ner_service)) -> ModelsResponse:
    """Return model registry information."""
    return service.list_models()


@router.post(
    "/activate",
    response_model=ModelsResponse,
    summary="Activate Model Version",
    description="Make a specific model version the default serving version.",
)
async def activate_model(
    request: ActivateModelRequest,
    service: NERService = Depends(get_ner_service),
) -> ModelsResponse:
    """Activate a model version."""
    try:
        service.activate_model(request.version_name)
        return service.list_models()
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))