"""Dependency registration and singleton initialization for Project 2."""

from typing import Optional

from fastapi import Depends, HTTPException, Request, Response, status

from app.config import settings
from app.services.cache_service import create_cache_backend
from app.services.rate_limiter import (
    FixedWindowRateLimiter,
    RateLimitExceeded,
    RateLimitStatus,
)
from app.services.sentiment_service import SentimentService


_sentiment_service: Optional[SentimentService] = None
_rate_limiter: Optional[FixedWindowRateLimiter] = None


def initialize_services() -> None:
    """Initialize singleton services once at application startup."""
    global _sentiment_service
    global _rate_limiter

    if _sentiment_service is not None and _sentiment_service.is_model_loaded():
        if _rate_limiter is None:
            _rate_limiter = FixedWindowRateLimiter(
                requests_per_window=settings.rate_limit_requests,
                window_seconds=settings.rate_limit_window_seconds,
            )
        return

    cache_backend = create_cache_backend(
        redis_url=settings.redis_url,
        default_ttl_seconds=settings.cache_ttl_seconds,
    )

    _sentiment_service = SentimentService(
        model_name=settings.sentiment_model_name,
        neutral_threshold=settings.neutral_threshold,
        cache_backend=cache_backend,
    )
    _sentiment_service.load_model(allow_degraded_startup=settings.allow_degraded_startup)

    _rate_limiter = FixedWindowRateLimiter(
        requests_per_window=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )


def get_sentiment_service() -> SentimentService:
    """Return the singleton sentiment service instance."""
    if _sentiment_service is None:
        raise RuntimeError(
            "SentimentService not initialized. Call initialize_services() at app startup."
        )

    return _sentiment_service


def get_rate_limiter() -> FixedWindowRateLimiter:
    """Return the singleton rate limiter instance."""
    if _rate_limiter is None:
        raise RuntimeError(
            "Rate limiter not initialized. Call initialize_services() at app startup."
        )

    return _rate_limiter


def enforce_rate_limit(
    request: Request,
    response: Response,
    limiter: FixedWindowRateLimiter = Depends(get_rate_limiter),
) -> None:
    """Apply per-client rate limiting to sentiment endpoints."""
    client_host = request.client.host if request.client else "anonymous"

    try:
        status_info: RateLimitStatus = limiter.check(client_host)
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please retry later.",
            headers={"Retry-After": str(exc.retry_after_seconds)},
        )

    response.headers["X-RateLimit-Limit"] = str(status_info.limit)
    response.headers["X-RateLimit-Remaining"] = str(status_info.remaining)
    response.headers["X-RateLimit-Reset"] = str(status_info.reset_after_seconds)