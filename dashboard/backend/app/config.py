"""
Configuration settings for the Dashboard Backend.

Uses pydantic-settings for environment variable management.
"""

import os
import secrets
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "Lemonade Eval Dashboard"
    app_version: str = "1.0.0"
    debug: bool = False

    # API settings
    api_v1_prefix: str = "/api/v1"
    allowed_hosts: list[str] = ["*"]
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description="Comma-separated list of allowed CORS origins",
    )

    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/lemonade_dashboard"
    database_async_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/lemonade_dashboard"

    # Security
    secret_key: str = Field(
        default_factory=lambda: os.environ.get("SECRET_KEY", ""),
        description="Secret key for JWT token generation - MUST be set in production",
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    api_key_prefix: str = "ledash_"

    # CLI Integration Security
    cli_secret: str = Field(
        default_factory=lambda: os.environ.get("CLI_SECRET", "dev-cli-secret-change-in-production"),
        description="Shared secret for CLI signature verification - MUST be set in production",
    )
    cli_signature_enabled: bool = Field(
        default=True,
        description="Enable/disable CLI signature verification (disable for development)",
    )

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v):
        """Validate that secret key is set and sufficiently long in production."""
        is_debug = os.environ.get("DEBUG", "false").lower() == "true"
        is_testing = os.environ.get("TESTING", "false").lower() == "true"

        # Skip validation in debug or testing mode
        if is_debug or is_testing:
            return v or secrets.token_urlsafe(32)

        # In production (non-debug mode), secret key must be set
        if not v or v == "your-secret-key-change-in-production":
            raise ValueError(
                "SECRET_KEY environment variable must be set in production. "
                "Generate a secure key with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
            )
        if len(v) < 32:
            raise ValueError(
                "SECRET_KEY must be at least 32 characters long in production"
            )
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string or list."""
        if isinstance(v, str):
            # Split comma-separated string
            origins = [origin.strip() for origin in v.split(",")]
            # Filter out empty strings and wildcards
            return [o for o in origins if o and o != "*"]
        return v

    # WebSocket
    ws_v1_prefix: str = "/ws/v1"

    # File storage
    cache_dir: Optional[str] = None  # Default lemonade cache directory

    # Redis (for caching and rate limiting)
    redis_url: str = "redis://localhost:6379/0"

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_default: int = 100  # requests per minute
    rate_limit_burst: int = 200  # max burst requests

    # Pagination
    default_page_size: int = 20
    max_page_size: int = 100


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
