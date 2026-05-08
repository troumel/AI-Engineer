"""Embedding providers for the RAG system."""

from __future__ import annotations

import hashlib
import importlib
from abc import ABC, abstractmethod
from math import sqrt


class EmbeddingProvider(ABC):
    """Abstract embedding provider used by the RAG service."""

    provider_name: str = "unknown"

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for one or more texts."""


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic local embedding fallback for development and tests."""

    provider_name = "hash-fallback"

    def __init__(self, dimensions: int = 96):
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_single(text) for text in texts]

    def _embed_single(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = [token.strip(".,!?;:\"'()[]{}") for token in text.lower().split() if token.strip()]

        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % self.dimensions
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            magnitude = 1.0 + (digest[3] / 255.0)
            vector[index] += sign * magnitude

        norm = sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings provider."""

    provider_name = "openai"

    def __init__(self, api_key: str, model_name: str):
        self.model_name = model_name
        openai_module = importlib.import_module("openai")
        client_type = getattr(openai_module, "OpenAI")
        self.client = client_type(api_key=api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in response.data]


def create_embedding_provider(api_key: str | None, model_name: str) -> EmbeddingProvider:
    """Create the configured embedding provider, falling back when needed."""
    if api_key:
        try:
            return OpenAIEmbeddingProvider(api_key=api_key, model_name=model_name)
        except Exception:
            return HashEmbeddingProvider()

    return HashEmbeddingProvider()