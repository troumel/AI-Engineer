"""Business logic for transformer-backed sentiment classification."""

from __future__ import annotations

import hashlib
import importlib
import logging
from typing import Any, Callable, Optional

from app.models.schemas import (
    BatchSentimentResponse,
    SentimentLabel,
    SentimentPrediction,
    SentimentResponse,
)
from app.services.cache_service import CacheBackend


logger = logging.getLogger(__name__)

ClassifierType = Callable[[str | list[str]], Any]


class SentimentService:
    """Load a HuggingFace sentiment pipeline and serve cached predictions."""

    def __init__(
        self,
        model_name: str,
        neutral_threshold: float,
        cache_backend: CacheBackend,
        classifier: Optional[ClassifierType] = None,
    ):
        self.model_name = model_name
        self.neutral_threshold = neutral_threshold
        self.cache_backend = cache_backend
        self.classifier: Optional[ClassifierType] = classifier

    @property
    def cache_backend_name(self) -> str:
        """Return the active cache backend name."""
        return self.cache_backend.backend_name

    def load_model(self, allow_degraded_startup: bool = True) -> None:
        """Load the HuggingFace pipeline unless a classifier is already injected."""
        if self.classifier is not None:
            return

        try:
            transformers = importlib.import_module("transformers")
            pipeline = getattr(transformers, "pipeline")

            self.classifier = pipeline(
                task="sentiment-analysis",
                model=self.model_name,
            )
            logger.info("Loaded sentiment model: %s", self.model_name)
        except Exception as exc:
            logger.warning("Unable to load sentiment model '%s': %s", self.model_name, exc)
            if not allow_degraded_startup:
                raise

    def is_model_loaded(self) -> bool:
        """Return whether a classifier backend is available."""
        return self.classifier is not None

    def predict(self, text: str) -> SentimentResponse:
        """Classify one text and cache the result."""
        cached_payload = self.cache_backend.get_json(self._build_cache_key(text))
        if cached_payload is not None:
            return SentimentResponse(**cached_payload, cached=True)

        if self.classifier is None:
            raise ValueError(
                "Sentiment model is not loaded. Install dependencies and verify HuggingFace access."
            )

        raw_result = self.classifier(text)
        normalized = self._normalize_result(text=text, raw_result=raw_result, cached=False)
        self.cache_backend.set_json(
            self._build_cache_key(text),
            normalized.model_dump(exclude={"cached"}),
        )
        return normalized

    def predict_batch(self, texts: list[str]) -> BatchSentimentResponse:
        """Classify multiple texts while reusing the single-item cache flow."""
        predictions: list[SentimentPrediction] = []
        cached_count = 0

        for text in texts:
            result = self.predict(text)
            if result.cached:
                cached_count += 1
            predictions.append(
                SentimentPrediction(
                    text=result.text,
                    label=result.label,
                    score=result.score,
                    cached=result.cached,
                )
            )

        return BatchSentimentResponse(
            model_name=self.model_name,
            total_texts=len(texts),
            cached_count=cached_count,
            predictions=predictions,
        )

    def _build_cache_key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"sentiment:{digest}"

    def _normalize_result(
        self,
        text: str,
        raw_result: Any,
        cached: bool,
    ) -> SentimentResponse:
        if isinstance(raw_result, list):
            item = raw_result[0]
        else:
            item = raw_result

        raw_label = str(item["label"]).strip().lower()
        score = float(item["score"])
        label = self._map_label(raw_label=raw_label, score=score)

        return SentimentResponse(
            text=text,
            label=label,
            score=score,
            model_name=self.model_name,
            cached=cached,
        )

    def _map_label(self, raw_label: str, score: float) -> SentimentLabel:
        normalized_label = raw_label.replace("label_1", "positive").replace(
            "label_0", "negative"
        )

        if score < self.neutral_threshold:
            return "neutral"

        if "positive" in normalized_label:
            return "positive"
        if "negative" in normalized_label:
            return "negative"

        return "neutral"