import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from chap_core.settings import DEFAULT_CORS_ORIGINS, ChapSettings, ConfigFileError, get_settings, load_dotenv_file


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


@pytest.mark.parametrize(("variable", "field"), [("CHAP_DEBUG", "debug"), ("IS_IN_DOCKER", "is_in_docker")])
@pytest.mark.parametrize(
    ("value", "expected"), [("YES", True), ("1", True), ("on", True), ("false", False), ("maybe", False)]
)
def test_flags_are_on_only_for_true_1_yes_or_on(settings_from, variable, field, value, expected):
    assert getattr(settings_from(**{variable: value}), field) is expected


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


@pytest.fixture
def config_file(monkeypatch, tmp_path):
    """Write a chap-server.yaml and point CHAP_CONFIG at it."""

    def write(text: str):
        path = tmp_path / "chap-server.yaml"
        path.write_text(text)
        monkeypatch.setenv("CHAP_CONFIG", str(path))
        return path

    return write


def test_the_config_file_fills_settings(config_file):
    config_file("cors_origins:\n  - https://dhis2.example.org/\nredis_port: 6380\nroot_path: /chap\n")

    settings = ChapSettings()

    assert settings.cors_origins == ["https://dhis2.example.org"]
    assert settings.redis_port == 6380
    assert settings.root_path == "/chap"


def test_an_environment_variable_overrides_the_config_file(config_file, monkeypatch):
    config_file("root_path: /from-file\nredis_port: 6380\n")
    monkeypatch.setenv("CHAP_ROOT_PATH", "/from-env")
    monkeypatch.setenv("REDIS_PORT", "6381")

    settings = ChapSettings()

    assert settings.root_path == "/from-env"
    assert settings.redis_port == 6381


def test_an_argument_overrides_the_environment_and_the_config_file(config_file, monkeypatch):
    config_file("root_path: /from-file\n")
    monkeypatch.setenv("CHAP_ROOT_PATH", "/from-env")

    assert ChapSettings(root_path="/from-argument").root_path == "/from-argument"


def test_the_config_file_overrides_the_defaults(config_file):
    config_file("marketplace_url: https://models.example.org\n")

    assert ChapSettings().marketplace_url == "https://models.example.org"


def test_no_file_is_read_without_chap_config(monkeypatch, tmp_path):
    monkeypatch.delenv("CHAP_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "chap-server.yaml").write_text("root_path: /ignored\n")

    assert ChapSettings().root_path == ""


def test_an_unknown_key_names_the_file_and_the_key(config_file):
    path = config_file("cors_orgins:\n  - https://dhis2.example.org\n")

    with pytest.raises(ConfigFileError, match=f"{path}: unknown setting\\(s\\) cors_orgins"):
        ChapSettings()


def test_a_missing_config_file_stops_startup(monkeypatch, tmp_path):
    monkeypatch.setenv("CHAP_CONFIG", str(tmp_path / "missing.yaml"))

    with pytest.raises(ConfigFileError, match="missing.yaml, which is not a file"):
        ChapSettings()


def test_an_empty_config_file_changes_nothing(config_file):
    config_file("")

    assert ChapSettings().root_path == ""


def test_the_example_config_file_is_valid(monkeypatch):
    example = Path(__file__).parent.parent / "chap-server.example.yaml"
    monkeypatch.setenv("CHAP_CONFIG", str(example))

    assert ChapSettings().cors_origins == ["https://dhis2.example.org"]


def test_a_key_given_twice_stops_startup_instead_of_replacing_the_first(config_file):
    path = config_file(
        "authentication:\n  tiers:\n    analyst:\n      permissions: [read:datasets]\n"
        "    analyst:\n      permissions: [delete:datasets]\n"
    )

    with pytest.raises(ConfigFileError, match=f"{path}: the key analyst appears twice \\(line 5\\)"):
        ChapSettings()


def test_yaml_merge_keys_still_work(config_file):
    config_file("<<: {debug: true, root_path: /merged}\nroot_path: /chap\n")

    settings = ChapSettings()

    assert settings.debug is True
    assert settings.root_path == "/chap"
