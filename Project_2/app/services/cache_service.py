"""Cache backends for sentiment inference results."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from threading import Lock
from typing import Any, Optional


class CacheBackend(ABC):
    """Abstract cache backend used by the sentiment service."""

    backend_name: str = "unknown"

    @abstractmethod
    def get_json(self, key: str) -> Optional[dict[str, Any]]:
        """Fetch a JSON-compatible object from the cache."""

    @abstractmethod
    def set_json(self, key: str, value: dict[str, Any]) -> None:
        """Store a JSON-compatible object in the cache."""


class InMemoryCacheBackend(CacheBackend):
    """Simple in-memory TTL cache for local development and tests."""

    backend_name = "memory"

    def __init__(self, default_ttl_seconds: int):
        self.default_ttl_seconds = default_ttl_seconds
        self._entries: dict[str, tuple[float, dict[str, Any]]] = {}
        self._lock = Lock()

    def get_json(self, key: str) -> Optional[dict[str, Any]]:
        now = time.time()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None

            expires_at, value = entry
            if expires_at <= now:
                self._entries.pop(key, None)
                return None

            return dict(value)

    def set_json(self, key: str, value: dict[str, Any]) -> None:
        expires_at = time.time() + self.default_ttl_seconds
        with self._lock:
            self._entries[key] = (expires_at, dict(value))


class RedisCacheBackend(CacheBackend):
    """Redis-backed cache implementation."""

    backend_name = "redis"

    def __init__(self, client: Any, default_ttl_seconds: int):
        self.client = client
        self.default_ttl_seconds = default_ttl_seconds

    def get_json(self, key: str) -> Optional[dict[str, Any]]:
        payload = self.client.get(key)
        if payload is None:
            return None

        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")

        return json.loads(payload)

    def set_json(self, key: str, value: dict[str, Any]) -> None:
        self.client.setex(key, self.default_ttl_seconds, json.dumps(value))


def create_cache_backend(redis_url: str, default_ttl_seconds: int) -> CacheBackend:
    """Create a Redis cache when available, otherwise fall back to memory."""
    try:
        import redis

        client = redis.Redis.from_url(redis_url, decode_responses=False)
        client.ping()
        return RedisCacheBackend(client=client, default_ttl_seconds=default_ttl_seconds)
    except Exception:
        return InMemoryCacheBackend(default_ttl_seconds=default_ttl_seconds)