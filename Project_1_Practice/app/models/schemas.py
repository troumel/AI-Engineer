"""
Pydantic schemas for data validation and serialization in Project_1_Practice.
"""

from pydantic import BaseModel, Field

class HealthCheckResponse(BaseModel):
    """
    Response model for health check endpoint.

    Indicates the service status and whether the ML model is loaded.
    """

    status: str = Field(
        ...,
        description="Service status",
        example="healthy"
    )

    model_loaded: bool = Field(
        ...,
        description="Indicates if the ML model is loaded",
        example=True
    )