"""OpenAI-compatible text-completion route."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import authenticate, get_llm_service
from app.models.schemas import CompletionRequest, CompletionResponse
from app.services.llm_service import (
    LLMService,
    ModelNotFoundError,
    RateLimitedError,
)

router = APIRouter(prefix="/v1", tags=["completions"])


@router.post("/completions", response_model=CompletionResponse)
async def create_completion(
    request: CompletionRequest,
    api_key: str = Depends(authenticate),
    llm_service: LLMService = Depends(get_llm_service),
) -> CompletionResponse:
    try:
        return await llm_service.create_completion(api_key, request)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RateLimitedError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
        ) from exc
