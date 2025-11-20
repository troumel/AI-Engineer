"""
Prediction service for ML model inference.

This is the business logic layer - similar to a service class in .NET.
Handles:
- Loading the ML model
- Making predictions
- Converting between API models and ML model formats
"""

import logging
from pathlib import Path
import numpy as np
import joblib

from app.models.schemas import PredictionRequest, PredictionResponse
from app.config import settings


# Set up logging
logger = logging.getLogger(__name__)


class PredictionService:
    """
    Service for loading the ML model and making predictions.

    Similar to a service class in .NET that you'd inject into controllers.
    """

    def __init__(self, model_path: str):
        """
        Initialize the prediction service.

        Args:
            model_path: Path to the trained model file (.joblib)
        """
        self.model_path = Path(model_path)
        self.model = None
        self.model_version = settings.app_version
        logger.info(f"PredictionService initialized with model path: {self.model_path}")

    def load_model(self):
        """
        Load the trained ML model from disk.

        This should be called once at application startup.
        Loading a 58MB model takes ~100ms, so we don't want to do this per request!

        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found at: {self.model_path}. "
                    f"Please train the model first by running: python scripts/train_model.py"
                )

            logger.info(f"Loading model from: {self.model_path}")

            # Load the model using joblib
            # This deserializes the RandomForestRegressor we saved during training
            self.model = joblib.load(self.model_path)

            logger.info("Model loaded successfully!")

        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def is_model_loaded(self) -> bool:
        """
        Check if the model is loaded and ready.

        Returns:
            bool: True if model is loaded, False otherwise
        """
        return self.model is not None

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Make a house price prediction.

        Args:
            request: PredictionRequest containing the 8 housing features

        Returns:
            PredictionResponse with predicted price

        Raises:
            ValueError: If model is not loaded
        """
        # Check if model is loaded
        if not self.is_model_loaded():
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            # Convert PredictionRequest to numpy array
            # The model expects features in this specific order:
            # [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
            features = np.array([[
                request.median_income,
                request.house_age,
                request.avg_rooms,
                request.avg_bedrooms,
                request.population,
                request.avg_occupancy,
                request.latitude,
                request.longitude
            ]])

            # Make prediction
            # model.predict() returns array like [4.526] (price in $100,000s)
            prediction = self.model.predict(features)

            # Convert prediction to actual dollar amount
            # Model predicts in units of $100,000, so multiply by 100,000
            predicted_price = float(prediction[0] * 100000)

            logger.info(f"Prediction made: ${predicted_price:,.2f}")

            # Return as PredictionResponse
            return PredictionResponse(
                predicted_price=predicted_price,
                model_version=self.model_version
            )

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
