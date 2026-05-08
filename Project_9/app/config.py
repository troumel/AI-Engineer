"""Application settings for the PyTorch training pipeline scaffold."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or a .env file."""

    app_name: str = "PyTorch Ticket Classifier API"
    app_version: str = "0.1.0"
    environment: str = "development"

    api_host: str = "0.0.0.0"
    api_port: int = 8008
    log_level: str = "INFO"

    models_directory: str = "models"
    default_model_version: str = "ticket_classifier_v1"
    base_model_name: str = "distilbert-base-uncased"
    allow_degraded_startup: bool = True

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


settings = Settings()
