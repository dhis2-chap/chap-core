"""Chap's runtime settings, read from environment variables.

This is the one place chap-core reads its configuration. The API, the Celery worker and the
CLI all call ``get_settings()``. Variables are named ``CHAP_<FIELD>``, except a few that
predate the prefix and keep their old names through an alias.
"""

import json
import os
from pathlib import Path
from typing import Annotated, Any

from dotenv import find_dotenv, load_dotenv
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# A local front-end dev server.
DEFAULT_CORS_ORIGINS = ["http://localhost:3000"]
DEFAULT_MARKETPLACE_URL = "https://raw.githubusercontent.com/dhis2-chap/chap-marketplace/main"
DEFAULT_MLFLOW_EXPERIMENT_NAME = "chap-backtests"


def load_dotenv_file(dotenv_path: str | None = None) -> None:
    """Copy a ``.env`` file into the process environment, leaving variables already set alone.

    Without a path, the file is found by searching up from chap-core's own folder. Its values
    land in ``os.environ``, so model subprocesses see them too.
    """
    load_dotenv(dotenv_path or find_dotenv())


load_dotenv_file()


def _alias(name: str, default: Any = None) -> Any:
    """A field read from a variable without the CHAP_ prefix, under the name it has always had."""
    return Field(default, validation_alias=name)


class ChapSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CHAP_",
        # The environment also holds variables for other tools (gunicorn, compose).
        extra="ignore",
        validate_by_name=True,
    )

    database_url: str | None = None
    debug: bool = False
    logs_dir: Path = Path("logs/")
    runs_dir: Path = Path("runs/")
    root_path: str = ""
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=lambda: list(DEFAULT_CORS_ORIGINS))
    api_token: SecretStr | None = None
    service_registration_key: SecretStr | None = _alias("SERVICEKIT_REGISTRATION_KEY")
    # The Chap instance the admin CLI talks to.
    url: str | None = None
    marketplace_url: str = DEFAULT_MARKETPLACE_URL

    celery_broker: str = _alias("CELERY_BROKER", "redis://localhost:6379")
    redis_host: str = _alias("REDIS_HOST", "localhost")
    redis_port: int = _alias("REDIS_PORT", 6379)
    redis_password: SecretStr | None = _alias("REDIS_PASSWORD")

    mlflow_tracking_uri: str | None = _alias("MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = _alias("MLFLOW_EXPERIMENT_NAME", DEFAULT_MLFLOW_EXPERIMENT_NAME)

    # Set by the container build and image.
    git_revision: str = _alias("GIT_REVISION", "")
    is_in_docker: bool = _alias("IS_IN_DOCKER", False)

    @field_validator("debug", mode="before")
    @classmethod
    def _parse_debug(cls, value: Any) -> Any:
        # Anything other than these words means off, rather than failing validation.
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return value

    @field_validator("is_in_docker", mode="before")
    @classmethod
    def _parse_is_in_docker(cls, value: Any) -> Any:
        # Any non-empty value means yes.
        if isinstance(value, str):
            return bool(value)
        return value

    @field_validator("api_token", "service_registration_key", mode="before")
    @classmethod
    def _empty_secret_is_unset(cls, value: Any) -> Any:
        return value or None

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: Any) -> Any:
        """Take a comma-separated string or a JSON list. An empty value falls back to the default."""
        if isinstance(value, str):
            value = json.loads(value) if value.strip().startswith("[") else value.split(",")
        if isinstance(value, list):
            origins = [str(origin).strip().rstrip("/") for origin in value if str(origin).strip()]
            return origins or list(DEFAULT_CORS_ORIGINS)
        return value


_cache: dict[frozenset[tuple[str, str]], ChapSettings] = {}


def get_settings() -> ChapSettings:
    """The settings for the current environment, parsed once and then reused.

    The cache is keyed on the process environment, so a variable set after startup is still
    seen, as it was when each module read its own variable.
    """
    key = frozenset(os.environ.items())
    settings = _cache.get(key)
    if settings is None:
        _cache.clear()
        settings = _cache[key] = ChapSettings()
    return settings
