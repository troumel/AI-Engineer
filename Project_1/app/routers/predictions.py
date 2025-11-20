"""
Predictions router.

Handles all endpoints related to house price predictions.
Similar to a PredictionsController in .NET.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from app.models.schemas import PredictionRequest, PredictionResponse
from app.services.prediction_service import PredictionService
from app.dependencies import get_prediction_service


# Create router for prediction endpoints
# All routes here will be under /predict
router = APIRouter(prefix="/predict", tags=["Predictions"])


@router.post(
    "",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict House Price",
    description="Predict the median house value based on housing features",
)
async def predict(
    request: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    """
    Predict house price endpoint.

    Takes housing features and returns predicted median house value.

    The request body should contain:
    - median_income: Median income in block (in $10,000s)
    - house_age: Median house age
    - avg_rooms: Average rooms per household
    - avg_bedrooms: Average bedrooms per household
    - population: Block population
    - avg_occupancy: Average household size
    - latitude: Block latitude
    - longitude: Block longitude

    Returns:
    - predicted_price: Predicted median house value in dollars
    - model_version: Version of the model used

    Args:
        request: PredictionRequest with the 8 housing features
        service: PredictionService injected via DI

    Returns:
        PredictionResponse with predicted price

    Raises:
        HTTPException: If prediction fails
    """
    try:
        # Call the service to make prediction
        # The service handles all the ML logic
        response = service.predict(request)

        return response

    except ValueError as e:
        # ValueError usually means model not loaded or invalid input
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}",
        )
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )
