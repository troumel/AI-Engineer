"""Application configuration for the basic RAG system."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or a .env file."""

    app_name: str = "Basic RAG API"
    app_version: str = "1.0.0"
    environment: str = "development"

    api_host: str = "0.0.0.0"
    api_port: int = 8002
    log_level: str = "INFO"

    upload_directory: str = "data/uploads"
    chroma_persist_directory: str = "data/chroma"
    chroma_collection_name: str = "project3_documents"
    vector_store_backend: str = "chroma"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 3
    max_upload_size_mb: int = 10

    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    allow_degraded_startup: bool = True

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


settings = Settings()