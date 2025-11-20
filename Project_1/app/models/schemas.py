"""
Pydantic models (DTOs) for request/response validation.

These are similar to C# DTOs with validation attributes, but Pydantic:
- Automatically validates types
- Provides clear error messages
- Generates JSON schema for API docs
- Handles serialization/deserialization
"""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Request model for house price prediction.

    Contains the 8 features from the California housing dataset.
    Similar to a C# class with [Required] and [Range] attributes.
    """

    median_income: float = Field(
        ...,  # ... means required (no default value)
        description="Median income in block group (in tens of thousands)",
        example=8.3252,
        gt=0  # Greater than 0 (validation rule)
    )

    house_age: float = Field(
        ...,
        description="Median house age in block group",
        example=41.0,
        ge=0  # Greater than or equal to 0
    )

    avg_rooms: float = Field(
        ...,
        description="Average number of rooms per household",
        example=6.98,
        gt=0
    )

    avg_bedrooms: float = Field(
        ...,
        description="Average number of bedrooms per household",
        example=1.02,
        gt=0
    )

    population: float = Field(
        ...,
        description="Block group population",
        example=322.0,
        gt=0
    )

    avg_occupancy: float = Field(
        ...,
        description="Average number of household members",
        example=2.55,
        gt=0
    )

    latitude: float = Field(
        ...,
        description="Block group latitude",
        example=37.88,
        ge=-90,  # Valid latitude range
        le=90
    )

    longitude: float = Field(
        ...,
        description="Block group longitude",
        example=-122.23,
        ge=-180,  # Valid longitude range
        le=180
    )

    class Config:
        """
        Pydantic configuration for this model.

        json_schema_extra: Provides example data for API documentation
        """
        json_schema_extra = {
            "example": {
                "median_income": 8.3252,
                "house_age": 41.0,
                "avg_rooms": 6.98,
                "avg_bedrooms": 1.02,
                "population": 322.0,
                "avg_occupancy": 2.55,
                "latitude": 37.88,
                "longitude": -122.23
            }
        }


class PredictionResponse(BaseModel):
    """
    Response model for house price prediction.

    Returns the predicted price and model information.
    """

    predicted_price: float = Field(
        ...,
        description="Predicted median house value in dollars",
        example=452600.0
    )

    model_version: str = Field(
        ...,
        description="Version of the model used for prediction",
        example="1.0.0"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_price": 452600.0,
                "model_version": "1.0.0"
            }
        }


class HealthCheckResponse(BaseModel):
    """
    Response model for health check endpoint.

    Indicates if the service is running and model is loaded.
    """

    status: str = Field(
        ...,
        description="Service status",
        example="healthy"
    )

    model_loaded: bool = Field(
        ...,
        description="Whether the ML model is loaded and ready",
        example=True
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True
            }
        }
