"""Application configuration for the sentiment analysis API."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or a .env file."""

    app_name: str = "Sentiment Analysis API"
    app_version: str = "1.0.0"
    environment: str = "development"

    api_host: str = "0.0.0.0"
    api_port: int = 8001

    log_level: str = "INFO"

    sentiment_model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    neutral_threshold: float = 0.70
    allow_degraded_startup: bool = True

    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 300

    rate_limit_requests: int = 30
    rate_limit_window_seconds: int = 60
    batch_max_size: int = 10

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


settings = Settings()