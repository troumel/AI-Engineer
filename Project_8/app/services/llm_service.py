"""High-level orchestration of model registry, batcher, usage, and rate limit."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import AsyncIterator, Iterable

from app.models.schemas import (
    ChatChoice,
    ChatChunkChoice,
    ChatChunkDelta,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    HealthCheckResponse,
    ModelInfo,
    ModelListResponse,
    Usage,
    UsageReport,
)
from app.services.batcher import ContinuousBatcher
from app.services.llm_engine import GenerationRequest, LLMEngine
from app.services.metrics import Metrics
from app.services.model_registry import ModelRegistry
from app.services.rate_limiter import RateLimiter
from app.services.tokenization import count_tokens
from app.services.usage_tracker import UsageTracker


class ModelNotFoundError(Exception):
    """Raised when an unknown `model` is requested."""


class RateLimitedError(Exception):
    """Raised when the rate limiter rejects a request."""


class LLMService:
    """Coordinate inference, batching, usage, metrics, and rate limiting."""

    def __init__(
        self,
        engine: LLMEngine,
        batcher: ContinuousBatcher,
        registry: ModelRegistry,
        rate_limiter: RateLimiter,
        usage_tracker: UsageTracker,
        metrics: Metrics,
        default_model: str,
        max_output_tokens: int,
    ) -> None:
        self.engine = engine
        self.batcher = batcher
        self.registry = registry
        self.rate_limiter = rate_limiter
        self.usage_tracker = usage_tracker
        self.metrics = metrics
        self.default_model = default_model
        self.max_output_tokens = max_output_tokens

    # ------------------------------------------------------------------
    # Models registry
    # ------------------------------------------------------------------

    def list_models(self) -> ModelListResponse:
        return ModelListResponse(data=self.registry.list())

    def get_model(self, model_id: str) -> ModelInfo:
        try:
            return self.registry.get(model_id)
        except KeyError as exc:
            raise ModelNotFoundError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def get_health(self) -> HealthCheckResponse:
        return HealthCheckResponse(
            status="healthy",
            backend=self.engine.name,
            default_model=self.default_model,
            registered_models=len(self.registry),
            queue_depth=self.batcher.queue_depth,
            max_batch_size=self.batcher.max_batch_size,
        )

    # ------------------------------------------------------------------
    # Chat completions
    # ------------------------------------------------------------------

    async def create_chat_completion(
        self,
        api_key: str,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        prompt = self._render_chat_prompt(request.messages)
        result, prompt_tokens, completion_tokens, latency = await self._run_generation(
            api_key=api_key,
            model=request.model,
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=self._normalise_stop(request.stop),
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=result.text),
                    finish_reason=self._coerce_finish(result.finish_reason),
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def stream_chat_completion(
        self,
        api_key: str,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[str]:
        self._validate_model(request.model)
        prompt = self._render_chat_prompt(request.messages)
        prompt_tokens = count_tokens(prompt)
        max_tokens = self._effective_max_tokens(request.max_tokens)
        self._enforce_rate_limit(api_key, prompt_tokens + max_tokens)

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        # Initial chunk announcing the assistant role.
        yield self._render_sse(
            ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    ChatChunkChoice(
                        index=0,
                        delta=ChatChunkDelta(role="assistant"),
                        finish_reason=None,
                    )
                ],
            )
        )

        gen_request = GenerationRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=self._normalise_stop(request.stop),
        )

        emitted_text_parts: list[str] = []
        emitted_tokens = 0
        start = time.perf_counter()

        for piece in self.engine.stream(gen_request):
            emitted_text_parts.append(piece)
            emitted_tokens += 1
            yield self._render_sse(
                ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatChunkChoice(
                            index=0,
                            delta=ChatChunkDelta(content=piece),
                            finish_reason=None,
                        )
                    ],
                )
            )
            # Yield to the event loop so consumers can iterate cooperatively.
            await asyncio.sleep(0)

        finish_reason = "length" if emitted_tokens >= max_tokens else "stop"
        yield self._render_sse(
            ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    ChatChunkChoice(
                        index=0,
                        delta=ChatChunkDelta(),
                        finish_reason=finish_reason,
                    )
                ],
            )
        )
        yield "data: [DONE]\n\n"

        latency = time.perf_counter() - start
        completion_tokens = max(emitted_tokens, count_tokens("".join(emitted_text_parts)))
        self.metrics.inc_request(prompt_tokens, completion_tokens, latency)
        self.usage_tracker.record(
            api_key=api_key,
            model=request.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    # ------------------------------------------------------------------
    # Text completions
    # ------------------------------------------------------------------

    async def create_completion(
        self,
        api_key: str,
        request: CompletionRequest,
    ) -> CompletionResponse:
        result, prompt_tokens, completion_tokens, _ = await self._run_generation(
            api_key=api_key,
            model=request.model,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=self._normalise_stop(request.stop),
        )
        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request.model,
            choices=[
                CompletionChoice(
                    index=0,
                    text=result.text,
                    finish_reason=self._coerce_finish(result.finish_reason),
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    # ------------------------------------------------------------------
    # Usage / metrics
    # ------------------------------------------------------------------

    def usage_report(self) -> UsageReport:
        return self.usage_tracker.report()

    def metrics_text(self) -> str:
        return self.metrics.render_prometheus()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _run_generation(
        self,
        api_key: str,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
    ):
        self._validate_model(model)
        prompt_tokens = count_tokens(prompt)
        bounded_max = self._effective_max_tokens(max_tokens)
        self._enforce_rate_limit(api_key, prompt_tokens + bounded_max)

        gen_request = GenerationRequest(
            prompt=prompt,
            max_tokens=bounded_max,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        start = time.perf_counter()
        try:
            result = await self.batcher.submit(gen_request)
        except Exception:
            self.metrics.inc_error()
            raise
        latency = time.perf_counter() - start

        completion_tokens = count_tokens(result.text)
        self.metrics.inc_request(prompt_tokens, completion_tokens, latency)
        self.usage_tracker.record(
            api_key=api_key,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return result, prompt_tokens, completion_tokens, latency

    def _validate_model(self, model_id: str) -> None:
        if not self.registry.has(model_id):
            raise ModelNotFoundError(f"Model '{model_id}' is not registered.")

    def _enforce_rate_limit(self, api_key: str, estimated_tokens: int) -> None:
        allowed, reason = self.rate_limiter.check(api_key, estimated_tokens)
        if not allowed:
            self.metrics.inc_error()
            raise RateLimitedError(reason or "rate limit exceeded")

    def _effective_max_tokens(self, requested: int) -> int:
        return min(max(requested, 1), self.max_output_tokens)

    @staticmethod
    def _render_chat_prompt(messages: Iterable[ChatMessage]) -> str:
        parts = []
        for message in messages:
            parts.append(f"<|{message.role}|>\n{message.content}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    @staticmethod
    def _normalise_stop(stop: list[str] | str | None) -> list[str] | None:
        if stop is None:
            return None
        if isinstance(stop, str):
            return [stop]
        return list(stop)

    @staticmethod
    def _render_sse(chunk: ChatCompletionChunk) -> str:
        return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

    @staticmethod
    def _coerce_finish(reason: str) -> str:
        return "length" if reason == "length" else "stop"
