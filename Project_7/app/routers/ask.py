"""Question answering routes (with streaming)."""

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.dependencies import get_rag_service
from app.models.schemas import AskRequest, AskResponse
from app.services.rag_service import AdvancedRagService

router = APIRouter(prefix="/ask", tags=["ask"])


@router.post("", response_model=AskResponse)
def ask(
    request: AskRequest,
    rag_service: AdvancedRagService = Depends(get_rag_service),
) -> AskResponse:
    return rag_service.ask(request)


@router.post("/stream")
def ask_stream(
    request: AskRequest,
    rag_service: AdvancedRagService = Depends(get_rag_service),
) -> StreamingResponse:
    return StreamingResponse(
        rag_service.stream_ask(request),
        media_type="text/event-stream",
    )
