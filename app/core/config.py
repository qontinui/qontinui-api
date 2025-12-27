"""Application configuration using pydantic-settings."""

import logging
import os
import warnings

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8001

    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost/qontinui"

    # API
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "Qontinui API"

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001"]

    # Security
    SECRET_KEY: str = "change-me-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_secret_key()

    def _validate_secret_key(self) -> None:
        """Validate SECRET_KEY is properly configured in production."""
        is_production = os.environ.get("ENVIRONMENT", "").lower() == "production"
        is_insecure = self.SECRET_KEY == "change-me-in-production"

        if is_production and is_insecure:
            raise RuntimeError(
                "CRITICAL SECURITY ERROR: SECRET_KEY must be set to a secure value in production. "
                "Set the SECRET_KEY environment variable to a strong random string."
            )
        elif is_insecure:
            warnings.warn(
                "SECURITY WARNING: Using default SECRET_KEY. "
                "Set SECRET_KEY environment variable for production use.",
                UserWarning,
                stacklevel=2,
            )
            logger.warning(
                "Using default SECRET_KEY - this is insecure for production. "
                "Set SECRET_KEY environment variable to a strong random string."
            )

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # Qontinui Web Backend (for historical data in integration testing)
    # This is the URL of the qontinui-web backend that stores historical execution data
    QONTINUI_WEB_URL: str = "http://localhost:8000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


settings = Settings()
