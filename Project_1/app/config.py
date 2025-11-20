"""
Application configuration using Pydantic Settings.

This is similar to appsettings.json + IOptions<T> in .NET, but more powerful:
- Type validation
- Environment variable loading
- Default values
- All in one place
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.

    Pydantic automatically:
    - Loads values from environment variables
    - Falls back to defaults if not set
    - Validates types
    - Converts values to correct types
    """

    # Application Info
    app_name: str = "House Price Prediction API"
    app_version: str = "1.0.0"
    environment: str = "development"  # development, staging, production

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Model Configuration
    model_path: str = "models/house_price_model.joblib"

    # Logging
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

    class Config:
        """
        Pydantic configuration.

        env_file: Load variables from .env file
        case_sensitive: Environment variable names are case-sensitive
        """
        env_file = ".env"
        case_sensitive = False


# Create a singleton instance
# This is like registering IOptions<Settings> in .NET
settings = Settings()
