"""Integration tests for the Project 2 API endpoints."""

import pytest
from fastapi.testclient import TestClient

from app import dependencies
from app.main import app
from app.services.cache_service import InMemoryCacheBackend
from app.services.rate_limiter import FixedWindowRateLimiter
from app.services.sentiment_service import SentimentService


class FakeClassifier:
    """Deterministic classifier used to avoid model downloads in tests."""

    def __call__(self, payload):
        if isinstance(payload, list):
            return [self._classify(text) for text in payload]
        return [self._classify(payload)]

    def _classify(self, text: str) -> dict[str, float | str]:
        lowered = text.lower()
        if any(word in lowered for word in ("love", "great", "excellent", "fast")):
            return {"label": "POSITIVE", "score": 0.98}
        if any(word in lowered for word in ("hate", "bad", "terrible", "slow")):
            return {"label": "NEGATIVE", "score": 0.97}
        return {"label": "POSITIVE", "score": 0.55}


@pytest.fixture(autouse=True)
def initialize_test_services():
    """Inject a fake classifier and relaxed rate limiter before each test."""
    dependencies._sentiment_service = SentimentService(
        model_name="test-sentiment-model",
        neutral_threshold=0.70,
        cache_backend=InMemoryCacheBackend(default_ttl_seconds=300),
        classifier=FakeClassifier(),
    )
    dependencies._sentiment_service.load_model()
    dependencies._rate_limiter = FixedWindowRateLimiter(
        requests_per_window=100,
        window_seconds=60,
    )
    yield
    dependencies._sentiment_service = None
    dependencies._rate_limiter = None


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_returns_api_info(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_check_returns_healthy(client):
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["cache_backend"] == "memory"


def test_sentiment_endpoint_returns_positive_result(client):
    response = client.post("/sentiment", json={"text": "I love this project."})

    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "positive"
    assert data["score"] == pytest.approx(0.98)
    assert data["cached"] is False


def test_sentiment_endpoint_uses_cache_on_second_request(client):
    first_response = client.post("/sentiment", json={"text": "I love this project."})
    second_response = client.post("/sentiment", json={"text": "I love this project."})

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json()["cached"] is False
    assert second_response.json()["cached"] is True


def test_sentiment_endpoint_returns_neutral_for_low_confidence(client):
    response = client.post("/sentiment", json={"text": "It is okay."})

    assert response.status_code == 200
    assert response.json()["label"] == "neutral"


def test_batch_sentiment_endpoint_returns_multiple_predictions(client):
    response = client.post(
        "/sentiment/batch",
        json={"texts": ["Great response time.", "This is terrible."]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total_texts"] == 2
    assert data["cached_count"] == 0
    assert [item["label"] for item in data["predictions"]] == ["positive", "negative"]


def test_batch_sentiment_counts_cached_predictions(client):
    client.post("/sentiment", json={"text": "Great response time."})

    response = client.post(
        "/sentiment/batch",
        json={"texts": ["Great response time.", "This is terrible."]},
    )

    assert response.status_code == 200
    assert response.json()["cached_count"] == 1


def test_sentiment_endpoint_rejects_blank_text(client):
    response = client.post("/sentiment", json={"text": "   "})

    assert response.status_code == 422


def test_batch_endpoint_rejects_empty_list(client):
    response = client.post("/sentiment/batch", json={"texts": []})

    assert response.status_code == 422


def test_rate_limiting_returns_429(client):
    dependencies._rate_limiter = FixedWindowRateLimiter(requests_per_window=1, window_seconds=60)

    first_response = client.post("/sentiment", json={"text": "I love this project."})
    second_response = client.post("/sentiment", json={"text": "I hate this latency."})

    assert first_response.status_code == 200
    assert second_response.status_code == 429
    assert second_response.headers["Retry-After"] == "60"