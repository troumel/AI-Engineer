"""Application configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings loaded from environment variables / .env file."""

    app_name: str = "Custom LLM Inference API"
    app_version: str = "1.0.0"
    environment: str = "development"

    api_host: str = "0.0.0.0"
    api_port: int = 8007
    log_level: str = "INFO"

    data_directory: str = "data"
    usage_file: str = "data/usage.json"

    llm_backend: str = "echo"
    default_model: str = "phi-3-mini-offline"

    max_batch_size: int = 8
    max_queue_depth: int = 64
    batch_timeout_ms: int = 20
    max_output_tokens: int = 256

    rate_limit_tokens_per_minute: int = 60000
    rate_limit_requests_per_minute: int = 120

    cost_per_1k_prompt_tokens: float = 0.0005
    cost_per_1k_completion_tokens: float = 0.0015

    api_keys: str = ""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    def parsed_api_keys(self) -> list[str]:
        return [key.strip() for key in self.api_keys.split(",") if key.strip()]


settings = Settings()
