"""Registry of served models with metadata (quantization, context window)."""

from __future__ import annotations

from threading import Lock

from app.models.schemas import ModelInfo


_DEFAULT_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="phi-3-mini-offline",
        owned_by="local",
        quantization="int4-awq",
        context_window=4096,
        max_output_tokens=512,
        backend="echo",
    ),
    ModelInfo(
        id="mistral-7b-offline",
        owned_by="local",
        quantization="int4-gptq",
        context_window=8192,
        max_output_tokens=512,
        backend="echo",
    ),
    ModelInfo(
        id="llama-3-8b-offline",
        owned_by="local",
        quantization="int8",
        context_window=8192,
        max_output_tokens=512,
        backend="echo",
    ),
]


class ModelRegistry:
    """Thread-safe registry of served models."""

    def __init__(self, models: list[ModelInfo] | None = None) -> None:
        self._models: dict[str, ModelInfo] = {}
        self._lock = Lock()
        for model in models or _DEFAULT_MODELS:
            self._models[model.id] = model

    def list(self) -> list[ModelInfo]:
        with self._lock:
            return list(self._models.values())

    def get(self, model_id: str) -> ModelInfo:
        with self._lock:
            model = self._models.get(model_id)
        if model is None:
            raise KeyError(f"Model '{model_id}' is not registered.")
        return model

    def has(self, model_id: str) -> bool:
        with self._lock:
            return model_id in self._models

    def register(self, model: ModelInfo) -> None:
        with self._lock:
            self._models[model.id] = model

    def __len__(self) -> int:
        with self._lock:
            return len(self._models)
