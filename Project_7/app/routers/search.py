"""Hybrid retrieval routes."""

from fastapi import APIRouter, Depends

from app.dependencies import get_rag_service
from app.models.schemas import SearchRequest, SearchResponse
from app.services.rag_service import AdvancedRagService

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(
    request: SearchRequest,
    rag_service: AdvancedRagService = Depends(get_rag_service),
) -> SearchResponse:
    return rag_service.search(request)
