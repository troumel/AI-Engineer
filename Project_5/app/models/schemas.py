"""Pydantic schemas for the multi-modal API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ImageInfo(BaseModel):
    """Metadata describing a stored image."""

    image_id: str
    filename: str
    content_type: str
    size_bytes: int
    created_at: str
    tags: list[str] = Field(default_factory=list)
    caption: Optional[str] = None


class ImageUploadResponse(ImageInfo):
    """Response payload returned after uploading an image."""

    storage_path: str


class ImageListResponse(BaseModel):
    """Response payload listing all stored images."""

    count: int
    images: list[ImageInfo]


class CaptionRequest(BaseModel):
    """Optional prompt used to bias the caption output."""

    prompt: Optional[str] = Field(
        default=None,
        description="Optional natural-language prompt to bias caption generation.",
    )


class CaptionResponse(BaseModel):
    """Generated caption for a stored image."""

    image_id: str
    caption: str
    model: str


class VQARequest(BaseModel):
    """Visual question answering request."""

    question: str = Field(min_length=1, max_length=500)


class VQAResponse(BaseModel):
    """Visual question answering response."""

    image_id: str
    question: str
    answer: str
    confidence: float
    model: str


class TextSearchRequest(BaseModel):
    """Search images by free-text description."""

    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=50)


class SearchResultItem(BaseModel):
    """A single result returned from a multi-modal similarity search."""

    image_id: str
    filename: str
    score: float
    caption: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """List of similarity search results."""

    query: str
    count: int
    results: list[SearchResultItem]


class ImageSearchRequest(BaseModel):
    """Find images visually similar to an existing stored image."""

    top_k: int = Field(default=5, ge=1, le=50)


class HealthCheckResponse(BaseModel):
    """Health-check payload."""

    status: str
    vision_provider: str
    image_count: int
    embedding_dimensions: int
