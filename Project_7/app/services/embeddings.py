"""Deterministic offline embedding provider.

The provider hashes tokens into a fixed-dimensional vector, then L2-normalises
the result. Production deployments would swap this out for OpenAI
`text-embedding-3-small` or `sentence-transformers`. Cosine similarity in
this space rewards token overlap, which is good enough for testing the
hybrid retrieval / re-ranking machinery without external dependencies.
"""

from __future__ import annotations

import hashlib
import math
from typing import Iterable

from app.services.text_utils import tokenize


class HashingEmbeddingProvider:
    """Hashing-based offline embedding."""

    name = "hashing"

    def __init__(self, dimensions: int = 128) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be >= 8.")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        return self._token_vector(tokenize(text, drop_stopwords=True))

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def _token_vector(self, tokens: list[str]) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokens:
            index = self._stable_index(token)
            vector[index] += 1.0
        norm = math.sqrt(sum(component * component for component in vector))
        if norm == 0:
            return vector
        return [component / norm for component in vector]

    def _stable_index(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % self.dimensions


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimensionality.")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
