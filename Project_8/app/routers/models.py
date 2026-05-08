"""OpenAI-compatible model registry routes."""

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import authenticate, get_llm_service
from app.models.schemas import ModelInfo, ModelListResponse
from app.services.llm_service import LLMService, ModelNotFoundError

router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models", response_model=ModelListResponse)
def list_models(
    _: str = Depends(authenticate),
    llm_service: LLMService = Depends(get_llm_service),
) -> ModelListResponse:
    return llm_service.list_models()


@router.get("/models/{model_id}", response_model=ModelInfo)
def get_model(
    model_id: str,
    _: str = Depends(authenticate),
    llm_service: LLMService = Depends(get_llm_service),
) -> ModelInfo:
    try:
        return llm_service.get_model(model_id)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
