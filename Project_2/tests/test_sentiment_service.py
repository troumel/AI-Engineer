"""Unit tests for the sentiment service."""

from app.services.cache_service import InMemoryCacheBackend
from app.services.sentiment_service import SentimentService


class FakeClassifier:
    """Classifier stub with deterministic output and call counting."""

    def __init__(self):
        self.call_count = 0

    def __call__(self, payload):
        self.call_count += 1
        if isinstance(payload, list):
            return [self._classify(text) for text in payload]
        return [self._classify(payload)]

    def _classify(self, text: str) -> dict[str, float | str]:
        lowered = text.lower()
        if "love" in lowered:
            return {"label": "POSITIVE", "score": 0.99}
        if "hate" in lowered:
            return {"label": "NEGATIVE", "score": 0.98}
        return {"label": "POSITIVE", "score": 0.60}


def test_predict_returns_positive_sentiment():
    service = SentimentService(
        model_name="test-model",
        neutral_threshold=0.70,
        cache_backend=InMemoryCacheBackend(default_ttl_seconds=300),
        classifier=FakeClassifier(),
    )

    response = service.predict("I love this API")

    assert response.label == "positive"
    assert response.score == 0.99
    assert response.cached is False


def test_predict_maps_low_confidence_to_neutral():
    service = SentimentService(
        model_name="test-model",
        neutral_threshold=0.70,
        cache_backend=InMemoryCacheBackend(default_ttl_seconds=300),
        classifier=FakeClassifier(),
    )

    response = service.predict("It is fine")

    assert response.label == "neutral"
    assert response.score == 0.60


def test_predict_uses_cache_for_repeated_text():
    classifier = FakeClassifier()
    service = SentimentService(
        model_name="test-model",
        neutral_threshold=0.70,
        cache_backend=InMemoryCacheBackend(default_ttl_seconds=300),
        classifier=classifier,
    )

    first_response = service.predict("I love caching")
    second_response = service.predict("I love caching")

    assert first_response.cached is False
    assert second_response.cached is True
    assert classifier.call_count == 1


def test_predict_batch_returns_expected_counts():
    service = SentimentService(
        model_name="test-model",
        neutral_threshold=0.70,
        cache_backend=InMemoryCacheBackend(default_ttl_seconds=300),
        classifier=FakeClassifier(),
    )

    service.predict("I love caching")
    response = service.predict_batch(["I love caching", "I hate waiting"])

    assert response.total_texts == 2
    assert response.cached_count == 1
    assert [item.label for item in response.predictions] == ["positive", "negative"]


def test_predict_raises_when_model_not_loaded():
    service = SentimentService(
        model_name="test-model",
        neutral_threshold=0.70,
        cache_backend=InMemoryCacheBackend(default_ttl_seconds=300),
    )

    try:
        service.predict("I love this API")
        raised = False
    except ValueError:
        raised = True

    assert raised is True