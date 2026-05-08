"""Pydantic request and response models for the sentiment API."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.config import settings


SentimentLabel = Literal["positive", "negative", "neutral"]


class SentimentRequest(BaseModel):
    """Single-text sentiment classification request."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to classify for sentiment.",
        examples=["I love how fast this API responds."],
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Text must not be blank.")
        return stripped


class BatchSentimentRequest(BaseModel):
    """Batch sentiment classification request."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=settings.batch_max_size,
        description="List of texts to classify in one request.",
        examples=[["Great product.", "This was disappointing."]],
    )

    @model_validator(mode="after")
    def validate_texts(self) -> "BatchSentimentRequest":
        cleaned: list[str] = []
        for text in self.texts:
            stripped = text.strip()
            if not stripped:
                raise ValueError("Batch texts must not contain blank values.")
            cleaned.append(stripped)
        self.texts = cleaned
        return self


class SentimentPrediction(BaseModel):
    """Sentiment classification result for one text."""

    text: str = Field(..., description="Original input text.")
    label: SentimentLabel = Field(..., description="Normalized sentiment label.")
    score: float = Field(..., ge=0.0, le=1.0, description="Model confidence score.")
    cached: bool = Field(..., description="Whether the result came from cache.")


class SentimentResponse(SentimentPrediction):
    """Response for single-text sentiment classification."""

    model_name: str = Field(..., description="HuggingFace model used for inference.")


class BatchSentimentResponse(BaseModel):
    """Response for batch sentiment classification."""

    model_name: str = Field(..., description="HuggingFace model used for inference.")
    total_texts: int = Field(..., ge=1, description="Number of input texts processed.")
    cached_count: int = Field(..., ge=0, description="Number of results served from cache.")
    predictions: list[SentimentPrediction] = Field(
        ..., description="Sentiment result for each input text."
    )


class HealthCheckResponse(BaseModel):
    """Health response with model and cache status."""

    status: str = Field(..., description="Service health status.")
    model_loaded: bool = Field(..., description="Whether the model is ready.")
    cache_backend: str = Field(..., description="Active cache backend name.")
    model_name: str = Field(..., description="Configured transformer model.")