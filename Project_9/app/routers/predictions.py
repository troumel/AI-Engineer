"""Prediction router for the ticket classifier."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_inference_service
from app.models.schemas import PredictionRequest, PredictionResponse
from app.services.inference_service import InferenceService

router = APIRouter(tags=["Predictions"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_label(
    request: PredictionRequest,
    service: InferenceService = Depends(get_inference_service),
) -> PredictionResponse:
    """Predict a ticket label for the provided text."""
    if not service.is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded.",
        )

    return service.predict(request.text)
