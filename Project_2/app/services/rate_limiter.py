"""Fixed-window rate limiter used by the sentiment endpoints."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class RateLimitStatus:
    """Current rate limit state for a client."""

    limit: int
    remaining: int
    reset_after_seconds: int


class RateLimitExceeded(Exception):
    """Raised when a client exceeds the configured limit."""

    def __init__(self, retry_after_seconds: int):
        super().__init__("Rate limit exceeded")
        self.retry_after_seconds = retry_after_seconds


class FixedWindowRateLimiter:
    """Thread-safe in-memory fixed-window limiter."""

    def __init__(self, requests_per_window: int, window_seconds: int):
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self._entries: dict[str, tuple[float, int]] = {}
        self._lock = Lock()

    def check(self, client_id: str) -> RateLimitStatus:
        """Record a request and return the remaining quota for the client."""
        now = time.time()

        with self._lock:
            window_started_at, request_count = self._entries.get(client_id, (now, 0))

            if now - window_started_at >= self.window_seconds:
                window_started_at = now
                request_count = 0

            if request_count >= self.requests_per_window:
                retry_after = max(1, math.ceil(self.window_seconds - (now - window_started_at)))
                raise RateLimitExceeded(retry_after_seconds=retry_after)

            request_count += 1
            self._entries[client_id] = (window_started_at, request_count)
            remaining = self.requests_per_window - request_count
            reset_after = max(0, math.ceil(self.window_seconds - (now - window_started_at)))

            return RateLimitStatus(
                limit=self.requests_per_window,
                remaining=remaining,
                reset_after_seconds=reset_after,
            )