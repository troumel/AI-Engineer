"""Application configuration for the NER API with fine-tuning."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or a .env file."""

    app_name: str = "NER Training API"
    app_version: str = "1.0.0"
    environment: str = "development"

    api_host: str = "0.0.0.0"
    api_port: int = 8003
    log_level: str = "INFO"

    models_directory: str = "models"
    default_model_version: str = "baseline-v1"
    base_model_name: str = "bert-base-cased"
    max_training_workers: int = 2

    default_candidate_percentage: float = 0.5
    allow_degraded_startup: bool = True

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


settings = Settings()