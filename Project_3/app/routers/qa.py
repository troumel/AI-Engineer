"""Question-answering endpoints for the RAG API."""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_rag_service
from app.models.schemas import AnswerResponse, QuestionRequest
from app.services.rag_service import RagService


router = APIRouter(prefix="/qa", tags=["Question Answering"])


@router.post(
    "/ask",
    response_model=AnswerResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask Question",
    description="Retrieve relevant context from uploaded documents and generate an answer with citations.",
)
async def ask_question(
    request: QuestionRequest,
    service: RagService = Depends(get_rag_service),
) -> AnswerResponse:
    """Answer a user question using retrieved document context."""
    try:
        return await asyncio.to_thread(
            service.answer_question,
            question=request.question,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {exc}",
        )