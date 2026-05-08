"""Pydantic schemas for the Project 9 inference API."""

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Text classification request."""

    text: str = Field(min_length=1, max_length=5000)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Text must not be blank.")
        return stripped


class PredictionResponse(BaseModel):
    """Text classification response."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    model_version: str
    backend: str


class HealthCheckResponse(BaseModel):
    """Health check payload."""

    status: str
    model_loaded: bool
    model_version: str
    base_model_name: str
    device: str
    backend: str
