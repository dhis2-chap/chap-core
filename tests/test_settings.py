import os

import pytest
from pydantic import ValidationError

from chap_core.settings import DEFAULT_CORS_ORIGINS, ChapSettings, get_settings, load_dotenv_file


@pytest.fixture
def settings_from(monkeypatch):
    """Settings read from the current environment with the given variables set."""

    def read(**variables: str) -> ChapSettings:
        for name, value in variables.items():
            monkeypatch.setenv(name, value)
        return ChapSettings()

    return read


def test_prefixed_variables_fill_their_fields(settings_from):
    settings = settings_from(CHAP_DATABASE_URL="postgresql://db/chap", CHAP_ROOT_PATH="/master")

    assert settings.database_url == "postgresql://db/chap"
    assert settings.root_path == "/master"


@pytest.mark.parametrize(
    ("variable", "field", "value", "expected"),
    [
        ("CELERY_BROKER", "celery_broker", "redis://broker:6379", "redis://broker:6379"),
        ("REDIS_HOST", "redis_host", "redis", "redis"),
        ("REDIS_PORT", "redis_port", "6380", 6380),
        ("MLFLOW_TRACKING_URI", "mlflow_tracking_uri", "file:///mlruns", "file:///mlruns"),
    ],
)
def test_variables_without_the_prefix_keep_their_names(settings_from, variable, field, value, expected):
    assert getattr(settings_from(**{variable: value}), field) == expected


def test_secrets_are_hidden_and_an_empty_one_is_unset(settings_from):
    settings = settings_from(CHAP_API_TOKEN="a" * 32, SERVICEKIT_REGISTRATION_KEY="")

    assert settings.api_token is not None
    assert settings.api_token.get_secret_value() == "a" * 32
    assert "a" * 32 not in repr(settings)
    assert settings.service_registration_key is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://a.example.org, https://b.example.org/,", ["https://a.example.org", "https://b.example.org"]),
        ('["https://a.example.org", "https://b.example.org/"]', ["https://a.example.org", "https://b.example.org"]),
        ("", DEFAULT_CORS_ORIGINS),
    ],
)
def test_cors_origins_take_a_comma_separated_string_or_a_json_list(settings_from, value, expected):
    assert settings_from(CHAP_CORS_ORIGINS=value).cors_origins == expected


@pytest.mark.parametrize(("value", "expected"), [("YES", True), ("1", True), ("false", False), ("maybe", False)])
def test_debug_is_on_only_for_true_1_or_yes(settings_from, value, expected):
    assert settings_from(CHAP_DEBUG=value).debug is expected


def test_an_invalid_value_fails_validation(settings_from):
    with pytest.raises(ValidationError, match="redis_port|REDIS_PORT"):
        settings_from(REDIS_PORT="not-a-port")


def test_a_dotenv_file_fills_the_environment_without_overriding_it(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setenv("REDIS_HOST", "from-environment")
    dotenv = tmp_path / ".env"
    dotenv.write_text("CELERY_BROKER=redis://from-dotenv:6379\nREDIS_HOST=from-dotenv\n")

    load_dotenv_file(str(dotenv))

    assert os.environ["CELERY_BROKER"] == "redis://from-dotenv:6379"
    assert get_settings().celery_broker == "redis://from-dotenv:6379"
    assert get_settings().redis_host == "from-environment"


def test_get_settings_is_reused_until_the_environment_changes(monkeypatch):
    first = get_settings()
    assert get_settings() is first

    monkeypatch.setenv("CHAP_ROOT_PATH", "/changed")

    assert get_settings().root_path == "/changed"
