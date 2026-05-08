"""Per-API-key token-bucket rate limiter."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock


@dataclass
class _Bucket:
    tokens: float
    requests: float
    updated_at: float


class RateLimiter:
    """Independent token-buckets for tokens-per-minute and requests-per-minute."""

    def __init__(
        self,
        tokens_per_minute: int,
        requests_per_minute: int,
    ) -> None:
        self.tokens_capacity = float(tokens_per_minute)
        self.requests_capacity = float(requests_per_minute)
        self.tokens_refill_rate = self.tokens_capacity / 60.0
        self.requests_refill_rate = self.requests_capacity / 60.0
        self._buckets: dict[str, _Bucket] = {}
        self._lock = Lock()

    def check(self, key: str, estimated_tokens: int) -> tuple[bool, str | None]:
        """Return `(allowed, reason)`. Consumes capacity when allowed."""
        if self.tokens_capacity <= 0 and self.requests_capacity <= 0:
            return True, None

        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(
                    tokens=self.tokens_capacity,
                    requests=self.requests_capacity,
                    updated_at=now,
                )
                self._buckets[key] = bucket
            else:
                elapsed = now - bucket.updated_at
                bucket.tokens = min(
                    self.tokens_capacity,
                    bucket.tokens + elapsed * self.tokens_refill_rate,
                )
                bucket.requests = min(
                    self.requests_capacity,
                    bucket.requests + elapsed * self.requests_refill_rate,
                )
                bucket.updated_at = now

            if self.requests_capacity > 0 and bucket.requests < 1.0:
                return False, "request rate limit exceeded"
            if self.tokens_capacity > 0 and bucket.tokens < estimated_tokens:
                return False, "token rate limit exceeded"

            if self.requests_capacity > 0:
                bucket.requests -= 1.0
            if self.tokens_capacity > 0:
                bucket.tokens -= float(estimated_tokens)
        return True, None
