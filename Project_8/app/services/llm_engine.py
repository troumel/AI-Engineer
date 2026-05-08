"""LLM engine abstraction and an offline echo-style backend.

The `LLMEngine` interface is intentionally narrow so production backends
(`vLLMEngine`, `TransformersEngine`, `LlamaCppEngine`, OpenAI proxy) can be
plugged in via `build_engine()`. The default `EchoLLMEngine` runs fully
offline — it produces deterministic completions by combining the prompt with
a templated response, which is enough to validate streaming, batching,
usage tracking, rate limiting, and the OpenAI-compatible response shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Protocol

from app.services.tokenization import tokenize_words


@dataclass
class GenerationRequest:
    """One prompt + sampling parameters."""

    prompt: str
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    stop: list[str] | None = None


@dataclass
class GenerationResult:
    """Completion + finish reason."""

    text: str
    finish_reason: str  # "stop" | "length"


class LLMEngine(Protocol):
    """Protocol every inference backend implements."""

    name: str

    def generate(self, request: GenerationRequest) -> GenerationResult: ...

    def generate_batch(
        self, requests: list[GenerationRequest]
    ) -> list[GenerationResult]: ...

    def stream(self, request: GenerationRequest) -> Iterator[str]: ...


class EchoLLMEngine:
    """Deterministic offline LLM engine.

    Generates responses by extracting salient words from the prompt and
    weaving them into a templated answer. Output length always respects
    `max_tokens` and `stop` sequences so the API surface mirrors a real
    backend like vLLM or `transformers.generate`.
    """

    name = "echo"

    _TEMPLATE_PREFIX = "Synthesised response based on your prompt:"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return self._render(request, words_iter=self._build_words(request.prompt))

    def generate_batch(
        self, requests: list[GenerationRequest]
    ) -> list[GenerationResult]:
        return [self.generate(request) for request in requests]

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        words = self._build_words(request.prompt)
        emitted = 0
        stop_sequences = list(request.stop or [])
        produced = ""
        for word in words:
            if emitted >= request.max_tokens:
                break
            piece = (" " if produced else "") + word
            if any(stop_word and stop_word in (produced + piece) for stop_word in stop_sequences):
                break
            produced += piece
            emitted += 1
            yield piece

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _render(
        self,
        request: GenerationRequest,
        words_iter: Iterable[str],
    ) -> GenerationResult:
        words = list(words_iter)
        truncated = words[: request.max_tokens]
        text = " ".join(truncated)
        if request.stop:
            text = self._apply_stop(text, request.stop)
        if len(words) > request.max_tokens:
            return GenerationResult(text=text, finish_reason="length")
        return GenerationResult(text=text, finish_reason="stop")

    @staticmethod
    def _apply_stop(text: str, stops: list[str]) -> str:
        cut = len(text)
        for stop_word in stops:
            if not stop_word:
                continue
            index = text.find(stop_word)
            if index != -1:
                cut = min(cut, index)
        return text[:cut].rstrip()

    def _build_words(self, prompt: str) -> list[str]:
        prompt_words = tokenize_words(prompt)
        salient = [word for word in prompt_words if len(word) > 3][:8]
        if not salient:
            salient = prompt_words[:4] or ["acknowledged"]
        body = " ".join(salient)
        full = f"{self._TEMPLATE_PREFIX} {body}. Answer is grounded in the prompt above."
        return full.split()


def build_engine(backend: str) -> LLMEngine:
    """Engine factory.

    The offline `echo` engine is the only fully-implemented backend in this
    project. The remaining branches show where a production deployment would
    plug in vLLM, HuggingFace Transformers, or llama.cpp.
    """
    backend = (backend or "echo").lower()
    if backend == "echo":
        return EchoLLMEngine()
    if backend in {"vllm", "transformers", "llamacpp"}:
        # Production slots — fall back to the offline engine until the
        # corresponding optional dependencies are installed and configured.
        return EchoLLMEngine()
    raise ValueError(f"Unknown LLM backend '{backend}'.")
