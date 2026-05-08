"""Sentiment classification endpoints."""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.dependencies import enforce_rate_limit, get_sentiment_service
from app.models.schemas import (
    BatchSentimentRequest,
    BatchSentimentResponse,
    SentimentRequest,
    SentimentResponse,
)
from app.services.sentiment_service import SentimentService


router = APIRouter(prefix="/sentiment", tags=["Sentiment"])


@router.post(
    "",
    response_model=SentimentResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify Sentiment",
    description="Classify a single text as positive, negative, or neutral.",
)
async def classify_sentiment(
    request: SentimentRequest,
    response: Response,
    _: None = Depends(enforce_rate_limit),
    service: SentimentService = Depends(get_sentiment_service),
) -> SentimentResponse:
    """Classify one text input."""
    try:
        return await asyncio.to_thread(service.predict, request.text)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {exc}",
        )


@router.post(
    "/batch",
    response_model=BatchSentimentResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify Batch Sentiment",
    description="Classify multiple texts in a single request.",
)
async def classify_batch_sentiment(
    request: BatchSentimentRequest,
    response: Response,
    _: None = Depends(enforce_rate_limit),
    service: SentimentService = Depends(get_sentiment_service),
) -> BatchSentimentResponse:
    """Classify a list of texts."""
    try:
        return await asyncio.to_thread(service.predict_batch, request.texts)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {exc}",
        )