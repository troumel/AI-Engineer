"""Image upload, retrieval, captioning, and visual question answering endpoints."""

import asyncio
import json
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from app.dependencies import get_multimodal_service
from app.models.schemas import (
    CaptionRequest,
    CaptionResponse,
    ImageInfo,
    ImageListResponse,
    ImageUploadResponse,
    VQARequest,
    VQAResponse,
)
from app.services.multimodal_service import MultiModalService


router = APIRouter(prefix="/images", tags=["Images"])


def _parse_tags(raw: Optional[str]) -> list[str]:
    """Accept tags as JSON array or comma-separated string."""
    if not raw:
        return []
    raw = raw.strip()
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON in tags: {exc}",
            )
        if not isinstance(parsed, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tags JSON must be an array of strings.",
            )
        return [str(item) for item in parsed]
    return [token.strip() for token in raw.split(",") if token.strip()]


@router.post(
    "",
    response_model=ImageUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload an image",
)
async def upload_image(
    file: UploadFile = File(...),
    tags: Optional[str] = Form(default=None),
    service: MultiModalService = Depends(get_multimodal_service),
) -> ImageUploadResponse:
    """Persist an uploaded image and add it to the multi-modal index."""
    parsed_tags = _parse_tags(tags)
    data = await file.read()
    try:
        return await asyncio.to_thread(
            service.upload_image,
            filename=file.filename or "upload.bin",
            content_type=file.content_type or "application/octet-stream",
            data=data,
            tags=parsed_tags,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get("", response_model=ImageListResponse, summary="List uploaded images")
async def list_images(
    service: MultiModalService = Depends(get_multimodal_service),
) -> ImageListResponse:
    """List all stored image records."""
    return await asyncio.to_thread(service.list_images)


@router.get(
    "/{image_id}",
    response_model=ImageInfo,
    summary="Get image metadata",
)
async def get_image(
    image_id: str,
    service: MultiModalService = Depends(get_multimodal_service),
) -> ImageInfo:
    """Return metadata for a stored image."""
    try:
        return await asyncio.to_thread(service.get_image, image_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.post(
    "/{image_id}/caption",
    response_model=CaptionResponse,
    summary="Generate a caption for an image",
)
async def caption_image(
    image_id: str,
    request: CaptionRequest = CaptionRequest(),
    service: MultiModalService = Depends(get_multimodal_service),
) -> CaptionResponse:
    """Run caption generation on the stored image."""
    try:
        return await asyncio.to_thread(
            service.caption_image,
            image_id=image_id,
            prompt=request.prompt,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.post(
    "/{image_id}/vqa",
    response_model=VQAResponse,
    summary="Visual question answering",
)
async def visual_question_answering(
    image_id: str,
    request: VQARequest,
    service: MultiModalService = Depends(get_multimodal_service),
) -> VQAResponse:
    """Answer a natural-language question about an image."""
    try:
        return await asyncio.to_thread(
            service.answer_question,
            image_id=image_id,
            question=request.question,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
