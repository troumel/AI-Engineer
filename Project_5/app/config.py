"""Application configuration for the multi-modal API."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or a .env file."""

    app_name: str = "Multi-Modal AI API"
    app_version: str = "1.0.0"
    environment: str = "development"

    api_host: str = "0.0.0.0"
    api_port: int = 8004
    log_level: str = "INFO"

    data_directory: str = "data"
    images_directory: str = "data/images"
    metadata_file: str = "data/images/metadata.json"
    max_upload_bytes: int = 10 * 1024 * 1024
    embedding_dimensions: int = 64

    vision_provider: str = "hashing"
    caption_model: str = "Salesforce/blip-image-captioning-base"
    clip_model: str = "openai/clip-vit-base-patch32"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


settings = Settings()
