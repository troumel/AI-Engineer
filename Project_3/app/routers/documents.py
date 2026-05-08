"""Document ingestion endpoints for the RAG API."""

import asyncio

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.config import settings
from app.dependencies import get_rag_service
from app.models.schemas import DocumentUploadResponse, DocumentsResponse
from app.services.rag_service import RagService


router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload Document",
    description="Upload a TXT or PDF file, split it into chunks, embed it, and store it for retrieval.",
)
async def upload_document(
    file: UploadFile = File(...),
    service: RagService = Depends(get_rag_service),
) -> DocumentUploadResponse:
    """Upload and index one document."""
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename is required.")

    content = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {settings.max_upload_size_mb} MB upload limit.",
        )

    try:
        return await asyncio.to_thread(
            service.ingest_document,
            filename=file.filename,
            content_type=file.content_type,
            data=content,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {exc}",
        )


@router.get(
    "",
    response_model=DocumentsResponse,
    summary="List Documents",
    description="List documents that have been indexed and are available for question answering.",
)
async def list_documents(service: RagService = Depends(get_rag_service)) -> DocumentsResponse:
    """Return indexed document metadata."""
    return await asyncio.to_thread(service.list_documents)