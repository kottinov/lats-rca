"""Application settings using Pydantic Settings.

This module defines the Settings class for managing application configuration,
following the "Robust Python" principle of using type-safe configuration with
validation.

Settings are loaded from environment variables and .env files, with support
for development, testing, and production environments.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    This class provides type-safe access to configuration values,
    with automatic validation and environment variable loading.

    All settings can be overridden via environment variables with
    the same name (case-insensitive). For example, OPENAI_API_KEY
    environment variable will override the openai_api_key setting.

    Attributes:
        openai_api_key: OpenAI API key for LLM calls
        tavily_api_key: Tavily API key for web search
        openai_model: OpenAI model to use (default: gpt-4o)
        lats_n_candidates: Number of candidates to generate at each node
        lats_max_depth: Maximum depth of the search tree
        lats_exploration_weight: UCT exploration weight (C parameter)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log output format (json or text)

    Example:
        >>> settings = Settings()
        >>> settings.openai_model
        'gpt-4o'

        >>> # Override from environment
        >>> import os
        >>> os.environ['OPENAI_MODEL'] = 'gpt-4o-mini'
        >>> settings = Settings()
        >>> settings.openai_model
        'gpt-4o-mini'
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore extra environment variables
    )

    openai_api_key: str = Field(
        ...,
        description="OpenAI API key",
        min_length=1,
    )

    tavily_api_key: str = Field(
        ...,
        description="Tavily API key for web search",
        min_length=1,
    )

    openai_model: str = Field(
        default="gpt-4o",
        description="OpenAI model to use for LLM calls",
    )

    lats_n_candidates: int = Field(
        default=5,
        description="Number of candidate nodes to generate at each step",
        ge=1,
        le=20,
    )

    lats_max_depth: int = Field(
        default=5,
        description="Maximum depth of the search tree",
        ge=1,
        le=20,
    )

    lats_exploration_weight: float = Field(
        default=1.41,  # sqrt(2)
        description="UCT exploration weight (C parameter)",
        ge=0.0,
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )

    log_format: Literal["json", "text"] = Field(
        default="json",
        description="Log output format (json for production, text for development)",
    )

    @field_validator("openai_model")
    @classmethod
    def validate_openai_model(cls, v: str) -> str:
        """Validate that the OpenAI model is supported.

        Args:
            v: Model name to validate

        Returns:
            Validated model name

        Raises:
            ValueError: If model is not supported
        """
        supported_models = {
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "claude-sonnet-4-5",
            "claude-opus-4-5",
            "claude-haiku-4-5",
        }
        if v not in supported_models:
            raise ValueError(
                f"Unsupported model '{v}'. "
                f"Supported models: {', '.join(sorted(supported_models))}"
            )
        return v

    @field_validator("openai_api_key", "tavily_api_key")
    @classmethod
    def validate_api_key_format(cls, v: str, info: object) -> str:
        """Validate API key format.

        Args:
            v: API key to validate
            info: Field validation info

        Returns:
            Validated API key

        Raises:
            ValueError: If API key format is invalid
        """
        if v.startswith("sk-test-") or v.startswith("tvly-test-"):
            return v

        field_name = str(getattr(info, "field_name", "api_key"))
        if "openai" in field_name and not v.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        if "tavily" in field_name and not v.startswith("tvly-"):
            raise ValueError("Tavily API key must start with 'tvly-'")

        return v

    def get_env_file_path(self) -> Path | None:
        """Get the path to the .env file if it exists.

        Returns:
            Path to .env file or None if not found
        """
        env_file = Path(".env")
        return env_file if env_file.exists() else None


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    This function uses LRU cache to ensure only one Settings instance
    is created, making configuration access efficient.

    Returns:
        Application settings instance

    Example:
        >>> settings = get_settings()
        >>> settings.openai_model
        'gpt-4o'
    """
    return Settings()  # type: ignore[call-arg]
