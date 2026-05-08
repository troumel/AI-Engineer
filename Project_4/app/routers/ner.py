"""Named entity extraction endpoints for Project 4."""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_ner_service
from app.models.schemas import ExtractRequest, ExtractResponse
from app.services.ner_service import NERService


router = APIRouter(prefix="/ner", tags=["NER"])


@router.post(
    "/extract",
    response_model=ExtractResponse,
    summary="Extract Named Entities",
    description="Extract entities from text using an explicit version, the active version, or the configured A/B rollout.",
)
async def extract_entities(
    request: ExtractRequest,
    service: NERService = Depends(get_ner_service),
) -> ExtractResponse:
    """Extract named entities from input text."""
    try:
        return await asyncio.to_thread(
            service.extract_entities,
            text=request.text,
            model_version=request.model_version,
            use_ab_test=request.use_ab_test,
            audience_id=request.audience_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {exc}",
        )