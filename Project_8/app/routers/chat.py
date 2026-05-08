"""OpenAI-compatible chat-completions routes (with streaming)."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.dependencies import authenticate, get_llm_service
from app.models.schemas import ChatCompletionRequest, ChatCompletionResponse
from app.services.llm_service import (
    LLMService,
    ModelNotFoundError,
    RateLimitedError,
)

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(authenticate),
    llm_service: LLMService = Depends(get_llm_service),
):
    if request.stream:
        try:
            llm_service.get_model(request.model)
        except ModelNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        return StreamingResponse(
            llm_service.stream_chat_completion(api_key, request),
            media_type="text/event-stream",
        )
    try:
        response: ChatCompletionResponse = await llm_service.create_chat_completion(api_key, request)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RateLimitedError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
        ) from exc
    return response
