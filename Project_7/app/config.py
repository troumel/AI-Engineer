"""Application configuration for the advanced RAG API."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or a .env file."""

    app_name: str = "Advanced RAG API"
    app_version: str = "1.0.0"
    environment: str = "development"

    api_host: str = "0.0.0.0"
    api_port: int = 8006
    log_level: str = "INFO"

    data_directory: str = "data"
    storage_file: str = "data/corpus.json"

    embedding_dimensions: int = 128
    chunk_size: int = 400
    chunk_overlap: int = 80

    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    rrf_k: int = 60
    reranker_enabled: bool = True

    answer_backend: str = "rule_based"
    openai_api_key: str = ""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


settings = Settings()
