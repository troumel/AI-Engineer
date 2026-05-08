"""Cross-modal similarity search endpoints."""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_multimodal_service
from app.models.schemas import (
    ImageSearchRequest,
    SearchResponse,
    TextSearchRequest,
)
from app.services.multimodal_service import MultiModalService


router = APIRouter(prefix="/search", tags=["Search"])


@router.post(
    "/text",
    response_model=SearchResponse,
    summary="Search images by text description",
)
async def search_by_text(
    request: TextSearchRequest,
    service: MultiModalService = Depends(get_multimodal_service),
) -> SearchResponse:
    """Return images whose embeddings best match the text query."""
    return await asyncio.to_thread(
        service.search_by_text,
        query=request.query,
        top_k=request.top_k,
    )


@router.post(
    "/images/{image_id}",
    response_model=SearchResponse,
    summary="Search images visually similar to an existing image",
)
async def search_by_image(
    image_id: str,
    request: ImageSearchRequest = ImageSearchRequest(),
    service: MultiModalService = Depends(get_multimodal_service),
) -> SearchResponse:
    """Return images visually similar to the supplied image id."""
    try:
        return await asyncio.to_thread(
            service.search_by_image,
            image_id=image_id,
            top_k=request.top_k,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
