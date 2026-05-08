"""Unit tests for the LLM service and supporting components."""

from __future__ import annotations

import asyncio

import pytest

from app.models.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
)
from app.services.batcher import ContinuousBatcher
from app.services.llm_engine import EchoLLMEngine, GenerationRequest
from app.services.llm_service import (
    LLMService,
    ModelNotFoundError,
    RateLimitedError,
)
from app.services.metrics import Metrics
from app.services.rate_limiter import RateLimiter
from app.services.tokenization import count_tokens


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def test_count_tokens_handles_empty_and_punctuation():
    assert count_tokens("") == 0
    assert count_tokens("hello") == 1
    assert count_tokens("hello, world!") >= 2


def test_echo_engine_respects_max_tokens_and_stop():
    engine = EchoLLMEngine()
    result = engine.generate(
        GenerationRequest(prompt="vector databases for retrieval augmented generation", max_tokens=4)
    )
    assert result.text
    assert len(result.text.split()) <= 4
    assert result.finish_reason == "length"


def test_echo_engine_stop_sequences_truncate_output():
    engine = EchoLLMEngine()
    result = engine.generate(
        GenerationRequest(
            prompt="quantize an LLM with bitsandbytes",
            max_tokens=64,
            stop=["grounded"],
        )
    )
    assert "grounded" not in result.text


def test_rate_limiter_blocks_after_exceeding_request_quota():
    limiter = RateLimiter(tokens_per_minute=10_000, requests_per_minute=2)
    assert limiter.check("k", 1) == (True, None)
    assert limiter.check("k", 1) == (True, None)
    allowed, reason = limiter.check("k", 1)
    assert not allowed
    assert "request" in (reason or "")


def test_rate_limiter_blocks_after_token_quota_exhausted():
    limiter = RateLimiter(tokens_per_minute=100, requests_per_minute=1000)
    assert limiter.check("k", 60)[0] is True
    allowed, reason = limiter.check("k", 60)
    assert not allowed
    assert "token" in (reason or "")


# ---------------------------------------------------------------------------
# Service-level tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_completion_returns_openai_shape(llm_service: LLMService):
    response = await llm_service.create_chat_completion(
        api_key="test-key",
        request=ChatCompletionRequest(
            model="phi-3-mini-offline",
            messages=[ChatMessage(role="user", content="explain quantization")],
            max_tokens=16,
        ),
    )
    assert response.id.startswith("chatcmpl-")
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens == (
        response.usage.prompt_tokens + response.usage.completion_tokens
    )
    assert response.choices[0].finish_reason in {"stop", "length"}


@pytest.mark.asyncio
async def test_completion_endpoint_records_usage(llm_service: LLMService):
    await llm_service.create_completion(
        api_key="alpha",
        request=CompletionRequest(
            model="phi-3-mini-offline",
            prompt="describe paged attention",
            max_tokens=8,
        ),
    )
    await llm_service.create_completion(
        api_key="beta",
        request=CompletionRequest(
            model="mistral-7b-offline",
            prompt="describe continuous batching",
            max_tokens=8,
        ),
    )
    report = llm_service.usage_report()
    assert report.total_requests == 2
    assert "alpha" in report.by_api_key
    assert "beta" in report.by_api_key
    assert "phi-3-mini-offline" in report.by_model
    assert "mistral-7b-offline" in report.by_model
    assert report.total_cost_usd > 0


@pytest.mark.asyncio
async def test_unknown_model_raises_not_found(llm_service: LLMService):
    with pytest.raises(ModelNotFoundError):
        await llm_service.create_completion(
            api_key="a",
            request=CompletionRequest(
                model="does-not-exist",
                prompt="hello",
                max_tokens=4,
            ),
        )


@pytest.mark.asyncio
async def test_rate_limit_propagates_to_service(tmp_path):
    from tests.conftest import _build_service  # type: ignore

    service = _build_service(tmp_path)
    service.rate_limiter = RateLimiter(tokens_per_minute=1_000_000, requests_per_minute=1)
    await service.create_completion(
        api_key="user",
        request=CompletionRequest(model="phi-3-mini-offline", prompt="hi", max_tokens=2),
    )
    with pytest.raises(RateLimitedError):
        await service.create_completion(
            api_key="user",
            request=CompletionRequest(model="phi-3-mini-offline", prompt="hi", max_tokens=2),
        )


@pytest.mark.asyncio
async def test_streaming_chat_emits_role_then_content_then_done(llm_service: LLMService):
    chunks: list[str] = []
    async for piece in llm_service.stream_chat_completion(
        api_key="streamer",
        request=ChatCompletionRequest(
            model="phi-3-mini-offline",
            messages=[ChatMessage(role="user", content="explain vllm continuous batching")],
            max_tokens=8,
            stream=True,
        ),
    ):
        chunks.append(piece)

    assert chunks[-1].strip() == "data: [DONE]"
    role_chunks = [c for c in chunks if '"role":"assistant"' in c]
    content_chunks = [c for c in chunks if '"content":' in c]
    finish_chunks = [c for c in chunks if '"finish_reason":' in c and "null" not in c]
    assert role_chunks, "expected an initial role chunk"
    assert content_chunks, "expected at least one content chunk"
    assert finish_chunks, "expected a terminal chunk with finish_reason"


@pytest.mark.asyncio
async def test_continuous_batcher_groups_concurrent_requests():
    metrics = Metrics()
    engine = EchoLLMEngine()
    batcher = ContinuousBatcher(
        engine=engine,
        max_batch_size=4,
        max_queue_depth=16,
        batch_timeout_ms=20,
        metrics=metrics,
    )
    await batcher.start()
    try:
        results = await asyncio.gather(
            *[
                batcher.submit(
                    GenerationRequest(prompt=f"prompt {i} continuous batching", max_tokens=8)
                )
                for i in range(4)
            ]
        )
    finally:
        await batcher.stop()

    assert all(result.text for result in results)
    avg_batch_section = metrics.render_prometheus()
    assert "llm_batch_size_avg" in avg_batch_section


def test_metrics_render_includes_expected_series():
    metrics = Metrics()
    metrics.inc_request(prompt_tokens=10, completion_tokens=5, latency=0.05)
    metrics.record_batch(2)
    text = metrics.render_prometheus()
    assert "llm_requests_total 1" in text
    assert "llm_tokens_in_total 10" in text
    assert "llm_tokens_out_total 5" in text
    assert "llm_batch_size_avg" in text
