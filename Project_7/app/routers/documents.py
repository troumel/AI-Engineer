"""Document ingestion / management routes."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_rag_service
from app.models.schemas import (
    DocumentListResponse,
    IngestRequest,
    IngestResponse,
)
from app.services.rag_service import AdvancedRagService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
def ingest_document(
    request: IngestRequest,
    rag_service: AdvancedRagService = Depends(get_rag_service),
) -> IngestResponse:
    try:
        return rag_service.ingest(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("", response_model=DocumentListResponse)
def list_documents(
    rag_service: AdvancedRagService = Depends(get_rag_service),
) -> DocumentListResponse:
    return rag_service.list_documents()


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
def delete_document(
    document_id: str,
    rag_service: AdvancedRagService = Depends(get_rag_service),
) -> dict[str, int | str]:
    try:
        removed = rag_service.remove_document(document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"document_id": document_id, "removed_chunks": removed}
