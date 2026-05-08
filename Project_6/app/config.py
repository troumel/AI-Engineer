"""Application configuration for the AI Agent API."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or a .env file."""

    app_name: str = "AI Agent API"
    app_version: str = "1.0.0"
    environment: str = "development"

    api_host: str = "0.0.0.0"
    api_port: int = 8005
    log_level: str = "INFO"

    planner_backend: str = "rule_based"
    max_agent_iterations: int = 6

    data_directory: str = "data"
    conversations_file: str = "data/conversations.json"
    sqlite_database: str = "data/agent.db"

    openai_api_key: str = ""
    anthropic_api_key: str = ""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


settings = Settings()
