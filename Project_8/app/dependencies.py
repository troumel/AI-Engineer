"""Dependency wiring + singleton initialisation."""

from __future__ import annotations

from typing import Optional

from fastapi import Header, HTTPException, status

from app.config import settings
from app.services.batcher import ContinuousBatcher
from app.services.llm_engine import build_engine
from app.services.llm_service import LLMService
from app.services.metrics import Metrics
from app.services.model_registry import ModelRegistry
from app.services.rate_limiter import RateLimiter
from app.services.usage_tracker import UsageTracker


_llm_service: Optional[LLMService] = None


async def initialize_services() -> None:
    global _llm_service

    if _llm_service is not None:
        await _llm_service.batcher.start()
        return

    metrics = Metrics()
    engine = build_engine(settings.llm_backend)
    registry = ModelRegistry()
    rate_limiter = RateLimiter(
        tokens_per_minute=settings.rate_limit_tokens_per_minute,
        requests_per_minute=settings.rate_limit_requests_per_minute,
    )
    usage_tracker = UsageTracker(
        usage_file=settings.usage_file,
        cost_per_1k_prompt_tokens=settings.cost_per_1k_prompt_tokens,
        cost_per_1k_completion_tokens=settings.cost_per_1k_completion_tokens,
    )
    batcher = ContinuousBatcher(
        engine=engine,
        max_batch_size=settings.max_batch_size,
        max_queue_depth=settings.max_queue_depth,
        batch_timeout_ms=settings.batch_timeout_ms,
        metrics=metrics,
    )
    await batcher.start()

    _llm_service = LLMService(
        engine=engine,
        batcher=batcher,
        registry=registry,
        rate_limiter=rate_limiter,
        usage_tracker=usage_tracker,
        metrics=metrics,
        default_model=settings.default_model,
        max_output_tokens=settings.max_output_tokens,
    )


async def shutdown_services() -> None:
    global _llm_service
    if _llm_service is None:
        return
    await _llm_service.batcher.stop()


def get_llm_service() -> LLMService:
    if _llm_service is None:
        raise RuntimeError(
            "LLMService not initialized. Call initialize_services() at app startup."
        )
    return _llm_service


def authenticate(authorization: Optional[str] = Header(default=None)) -> str:
    """Return the API key associated with this request (or `anonymous`)."""
    configured = settings.parsed_api_keys()
    if not configured:
        # Auth disabled; bucket usage by header value when present.
        if authorization and authorization.lower().startswith("bearer "):
            return authorization.split(" ", 1)[1].strip() or "anonymous"
        return "anonymous"

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
        )
    token = authorization.split(" ", 1)[1].strip()
    if token not in configured:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
    return token
