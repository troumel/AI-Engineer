"""
Dependency injection for the application.

This is similar to registering services in Program.cs in .NET:
    builder.Services.AddSingleton<IPredictionService, PredictionService>();

FastAPI uses a different approach:
- Define a function that returns the dependency
- Use Depends() in route parameters to inject it
"""

from app.services.prediction_service import PredictionService
from app.config import settings


# Global singleton instance of the prediction service
# Created once when the app starts, reused for all requests
# Similar to registering as Singleton in .NET DI
_prediction_service: PredictionService = None


def get_prediction_service() -> PredictionService:
    """
    Dependency provider for PredictionService.

    Returns the singleton instance of PredictionService.
    This is called by FastAPI's dependency injection system.

    Usage in routes:
        @router.post("/predict")
        async def predict(
            request: PredictionRequest,
            service: PredictionService = Depends(get_prediction_service)
        ):
            return service.predict(request)

    Returns:
        PredictionService: The singleton prediction service instance
    """
    global _prediction_service

    # Return the existing instance
    # (It will be created at app startup via initialize_services())
    if _prediction_service is None:
        raise RuntimeError(
            "PredictionService not initialized. "
            "Call initialize_services() at app startup."
        )

    return _prediction_service


def initialize_services():
    """
    Initialize all singleton services at application startup.

    This should be called once when the FastAPI app starts.
    Similar to configuring services in .NET's Program.cs.

    This function:
    1. Creates the PredictionService instance
    2. Loads the ML model from disk
    3. Stores it as a global singleton
    """
    global _prediction_service

    # Create the service instance
    _prediction_service = PredictionService(model_path=settings.model_path)

    # Load the model (this happens once at startup, not per request!)
    _prediction_service.load_model()

    print(f"Services initialized successfully!")
    print(f"Model loaded from: {settings.model_path}")
