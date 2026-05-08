"""Vision provider abstractions with an offline hashing-based default.

The hashing provider produces deterministic embeddings, captions, and VQA
answers from the raw image bytes plus optional textual hints (tags/filename).
This keeps the project runnable and testable without downloading BLIP/CLIP
models, while preserving an obvious upgrade path: simply add a HuggingFace
provider that swaps `embed_image` / `caption` / `answer` with real models.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, Protocol


_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercased alphanumeric tokens extracted from arbitrary text."""
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text or "")]


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0:
        return vector
    return [component / norm for component in vector]


class VisionProvider(Protocol):
    """Capabilities required from any vision backend."""

    name: str

    def embed_text(self, text: str) -> list[float]:
        ...

    def embed_image(self, image_bytes: bytes, hints: Iterable[str]) -> list[float]:
        ...

    def caption(self, image_bytes: bytes, hints: Iterable[str], prompt: str | None) -> str:
        ...

    def answer(
        self,
        image_bytes: bytes,
        question: str,
        hints: Iterable[str],
    ) -> tuple[str, float]:
        ...


class HashingVisionProvider:
    """Deterministic provider that mixes image hashes with textual hints.

    The embedding space is shared between text and image so that text queries
    align with images whose hints overlap with the query terms. Image bytes
    contribute a stable per-image signature so visually identical images embed
    to the same vector regardless of hints.
    """

    name = "hashing"

    def __init__(self, dimensions: int = 64) -> None:
        if dimensions < 8:
            raise ValueError("Embedding dimensions must be >= 8.")
        self.dimensions = dimensions

    def embed_text(self, text: str) -> list[float]:
        return _normalize(self._token_vector(_tokenize(text)))

    def embed_image(self, image_bytes: bytes, hints: Iterable[str]) -> list[float]:
        hint_tokens = self._collect_hint_tokens(hints)
        text_component = self._token_vector(hint_tokens)
        image_component = self._image_signature(image_bytes)

        combined = [
            text_component[index] + 0.25 * image_component[index]
            for index in range(self.dimensions)
        ]
        return _normalize(combined)

    def caption(
        self,
        image_bytes: bytes,
        hints: Iterable[str],
        prompt: str | None,
    ) -> str:
        tokens = self._collect_hint_tokens(hints)
        prompt_text = (prompt or "").strip()
        digest = hashlib.sha1(image_bytes).hexdigest()[:8]

        if tokens:
            unique_tokens = list(dict.fromkeys(tokens))[:5]
            subject = ", ".join(unique_tokens)
            base = f"An image showing {subject}"
        else:
            base = f"An image (signature {digest})"

        if prompt_text:
            return f"{base}. {prompt_text.capitalize()}".strip()
        return f"{base}."

    def answer(
        self,
        image_bytes: bytes,
        question: str,
        hints: Iterable[str],
    ) -> tuple[str, float]:
        question_tokens = set(_tokenize(question))
        hint_tokens = set(self._collect_hint_tokens(hints))
        if not question_tokens:
            return "I cannot answer that question.", 0.1

        overlap = question_tokens & hint_tokens
        if overlap:
            joined = ", ".join(sorted(overlap))
            confidence = min(1.0, 0.5 + 0.1 * len(overlap))
            return f"Yes, the image relates to {joined}.", confidence

        digest_value = int(hashlib.sha1(image_bytes).hexdigest()[:4], 16)
        confidence = 0.2 + (digest_value % 30) / 100.0
        return "I am not certain based on the image content.", confidence

    def _token_vector(self, tokens: Iterable[str]) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokens:
            index = self._stable_index(f"tok:{token}")
            vector[index] += 1.0
        return vector

    def _image_signature(self, image_bytes: bytes) -> list[float]:
        digest = hashlib.sha256(image_bytes).digest()
        vector = [0.0] * self.dimensions
        for byte_index, byte_value in enumerate(digest):
            vector[byte_index % self.dimensions] += (byte_value / 255.0)
        return vector

    def _stable_index(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % self.dimensions

    def _collect_hint_tokens(self, hints: Iterable[str]) -> list[str]:
        tokens: list[str] = []
        for hint in hints:
            tokens.extend(_tokenize(hint))
        return tokens


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return the cosine similarity between two equal-length vectors."""
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimensionality.")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def build_vision_provider(name: str, dimensions: int) -> VisionProvider:
    """Return a vision provider instance for the configured backend.

    Only the offline hashing provider is wired in by default. Selecting the
    "huggingface" backend will fall back to the hashing provider when the
    optional `transformers` / `torch` / `Pillow` packages are not installed,
    so unit tests stay deterministic without those heavy dependencies.
    """
    normalized = (name or "hashing").lower()
    if normalized in {"hashing", "fake", "offline", "default"}:
        return HashingVisionProvider(dimensions=dimensions)

    if normalized in {"huggingface", "hf", "blip", "clip"}:
        try:  # pragma: no cover - exercised only when full ML stack is present
            from app.services.huggingface_vision import HuggingFaceVisionProvider

            return HuggingFaceVisionProvider(dimensions=dimensions)
        except Exception:
            return HashingVisionProvider(dimensions=dimensions)

    raise ValueError(f"Unsupported vision provider '{name}'.")
