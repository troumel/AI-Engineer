"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app import dependencies  # noqa: E402
from app.main import app  # noqa: E402
from app.services.batcher import ContinuousBatcher  # noqa: E402
from app.services.llm_engine import build_engine  # noqa: E402
from app.services.llm_service import LLMService  # noqa: E402
from app.services.metrics import Metrics  # noqa: E402
from app.services.model_registry import ModelRegistry  # noqa: E402
from app.services.rate_limiter import RateLimiter  # noqa: E402
from app.services.usage_tracker import UsageTracker  # noqa: E402


def _build_service(tmp_path) -> LLMService:
    metrics = Metrics()
    engine = build_engine("echo")
    registry = ModelRegistry()
    rate_limiter = RateLimiter(tokens_per_minute=60_000, requests_per_minute=120)
    usage_tracker = UsageTracker(
        usage_file=str(tmp_path / "usage.json"),
        cost_per_1k_prompt_tokens=0.0005,
        cost_per_1k_completion_tokens=0.0015,
    )
    batcher = ContinuousBatcher(
        engine=engine,
        max_batch_size=4,
        max_queue_depth=32,
        batch_timeout_ms=5,
        metrics=metrics,
    )
    return LLMService(
        engine=engine,
        batcher=batcher,
        registry=registry,
        rate_limiter=rate_limiter,
        usage_tracker=usage_tracker,
        metrics=metrics,
        default_model="phi-3-mini-offline",
        max_output_tokens=64,
    )


@pytest.fixture
def llm_service(tmp_path) -> LLMService:
    return _build_service(tmp_path)


@pytest.fixture
def client(tmp_path):
    service = _build_service(tmp_path)
    previous = dependencies._llm_service
    dependencies._llm_service = service
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        dependencies._llm_service = previous
