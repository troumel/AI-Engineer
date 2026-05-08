"""Offline RAG evaluation routes."""

from fastapi import APIRouter, Depends

from app.dependencies import get_rag_service
from app.models.schemas import EvaluationRequest, EvaluationResponse
from app.services.rag_service import AdvancedRagService

router = APIRouter(prefix="/evaluate", tags=["evaluate"])


@router.post("", response_model=EvaluationResponse)
def evaluate(
    request: EvaluationRequest,
    rag_service: AdvancedRagService = Depends(get_rag_service),
) -> EvaluationResponse:
    return rag_service.evaluate(request)
