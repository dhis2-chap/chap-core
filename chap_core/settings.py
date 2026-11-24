"""Chap's runtime settings, read from environment variables and an optional ``chap-server.yml``.

This is the one place chap-core reads its configuration. The API, the Celery worker and the
CLI all call ``get_settings()``. Variables are named ``CHAP_<FIELD>``, except a few that
predate the prefix and keep their old names through an alias.

``CHAP_CONFIG`` names a YAML file whose keys are the field names, such as ``cors_origins``.
Environment variables override the file, so a container can change one value without editing
it. Without ``CHAP_CONFIG`` no file is read.
"""

import json
import os
from pathlib import Path
from typing import Annotated, Any

import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import BeforeValidator, Field, SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from chap_core.auth.config import AuthenticationConfig

# A local front-end dev server.
DEFAULT_CORS_ORIGINS = ["http://localhost:3000"]
DEFAULT_MARKETPLACE_URL = "https://raw.githubusercontent.com/dhis2-chap/chap-marketplace/main"
DEFAULT_MLFLOW_EXPERIMENT_NAME = "chap-backtests"
CONFIG_ENV_VAR = "CHAP_CONFIG"


class ConfigFileError(ValueError):
    """``chap-server.yml`` is missing or holds a key Chap does not know."""


def load_dotenv_file(dotenv_path: str | None = None) -> None:
    """Copy a ``.env`` file into the process environment, leaving variables already set alone.

    Without a path, the file is found by searching up from chap-core's own folder. Its values
    land in ``os.environ``, so model subprocesses see them too. Only the API and the worker
    call this, through ``celery_tasks``, so the CLI and library use never pick up a ``.env``.
    """
    load_dotenv(dotenv_path or find_dotenv())


def parse_flag(value: Any) -> Any:
    """An on/off variable is on for ``true``, ``1``, ``yes`` or ``on`` (any case), and off for anything else."""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return value


# A bool that never fails validation on an unexpected word, unlike pydantic's own parsing.
Flag = Annotated[bool, BeforeValidator(parse_flag)]


def _alias(name: str, default: Any = None) -> Any:
    """A field read from a variable without the CHAP_ prefix, under the name it has always had."""
    return Field(default, validation_alias=name)


class ChapSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CHAP_",
        # Catches a typo in chap-server.yml. Unrelated environment variables are never read.
        extra="forbid",
        validate_by_name=True,
        # A secret pasted into the wrong setting must not be echoed in the startup error. Nested
        # models format their errors with this setting, not their own.
        hide_input_in_errors=True,
    )

    database_url: str | None = None
    debug: Flag = False
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
    is_in_docker: Flag = _alias("IS_IN_DOCKER", False)

    # Named tokens; see docs/contributor/auth_contract.md. Usually set in chap-server.yml.
    authentication: AuthenticationConfig = Field(default_factory=AuthenticationConfig)

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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Highest first. .env is copied into the environment (load_dotenv_file), so it ranks with it.
        config_path = os.environ.get(CONFIG_ENV_VAR)
        if not config_path:
            return init_settings, env_settings
        return init_settings, env_settings, ServerConfigFileSource(settings_cls, Path(config_path))


class _RefuseDuplicateKeys(yaml.SafeLoader):
    """YAML keeps the last of two equal keys, so a second ``analyst:`` tier would silently replace the first."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen = set()
        for key_node, _ in node.value:
            # A merge key (<<) is flattened by SafeLoader below; a key written next to it may
            # override a merged one, as YAML intends. Only keys written twice are refused.
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=deep)  # type: ignore[no-untyped-call]
            if key in seen:
                raise ConfigFileError(f"{self.name}: the key {key} appears twice (line {key_node.start_mark.line + 1})")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


class ServerConfigFileSource(YamlConfigSettingsSource):
    """``chap-server.yml``, failing with the file's path when it is missing, repeats a key or has an unknown one."""

    def __init__(self, settings_cls: type[BaseSettings], path: Path):
        if not path.is_file():
            raise ConfigFileError(f"{CONFIG_ENV_VAR} points to {path}, which is not a file")
        with path.open() as stream:
            yaml.load(stream, Loader=_RefuseDuplicateKeys)  # noqa: S506 — a SafeLoader subclass
        super().__init__(settings_cls, yaml_file=path)
        known = set(settings_cls.model_fields)
        unknown = sorted(set(self.yaml_data) - known)
        if unknown:
            raise ConfigFileError(f"{path}: unknown setting(s) {', '.join(unknown)}")


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
