
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import chap_core.database.database as db
from chap_core.database.model_templates_and_config_tables import ModelTemplateDB
from chap_core.exceptions import ModelTemplateRevisionConflict
from chap_core.models.builtin import BUILTIN_SOURCE_PREFIX


@pytest.mark.parametrize(
    "name, source_url, loaded_name, ignore_env, should_fail",
    [
        ("alias", "https://github.com/example/model@abc123", "original", False, False),
        ("chap_ewars_test", "https://github.com/example/model@abc123", "original", True, False),
        ("builtin_model", f"{BUILTIN_SOURCE_PREFIX}builtin_model", "wrong_name", False, True),
    ],
)
def test_get_model_template_with_code_non_chapkit(
    monkeypatch, name, source_url, loaded_name, ignore_env, should_fail
):
    wrapper = db.SessionWrapper()
    stored = ModelTemplateDB(name=name, version="v1", source_url=source_url)
    monkeypatch.setattr(wrapper, "get_model_template", lambda _: stored)
    executable = SimpleNamespace(
        model_template_config=SimpleNamespace(name=loaded_name, version="v1")
    )
    loader = Mock(return_value=executable)
    monkeypatch.setattr(db.ModelTemplate, "from_directory_or_github_url", loader)
    if should_fail:
        with pytest.raises(ValueError, match="does not match the stored template identity"):
            wrapper.get_model_template_with_code(1)
    else:
        assert wrapper.get_model_template_with_code(1) is executable
    loader.assert_called_once_with(source_url, ignore_env=ignore_env)


@pytest.mark.parametrize(
    "reported_name, reported_version, reported_digest, expected_error",
    [
        ("service", "v1", "abc123", None),
        ("service", "v1", "different", ModelTemplateRevisionConflict),
        ("service", "v2", "abc123", ValueError),
    ],
)
def test_get_model_template_with_code_chapkit(
    monkeypatch, reported_name, reported_version, reported_digest, expected_error
):
    wrapper = db.SessionWrapper()
    stored = ModelTemplateDB(
        name="service",
        version="v1",
        source_url="http://old-service",
        uses_chapkit=True,
        source_digest="abc123",
    )
    monkeypatch.setattr(wrapper, "get_model_template", lambda _: stored)

    resolver = Mock(return_value="http://live-service")
    monkeypatch.setattr(wrapper, "_resolve_chapkit_live_source_url", resolver)
    executable = Mock()
    executable.get_model_template_config_with_digest.return_value = (
        SimpleNamespace(name=reported_name, version=reported_version),
        reported_digest,
    )
    constructor = Mock(return_value=executable)
    monkeypatch.setattr(db, "ExternalChapkitModelTemplate", constructor)

    if expected_error:
        with pytest.raises(expected_error):
            wrapper.get_model_template_with_code(1)
        executable.close.assert_called_once_with()
    else:
        assert wrapper.get_model_template_with_code(1) is executable
        executable.close.assert_not_called()
    resolver.assert_called_once_with(
        service_id="service",
        stored_source_url="http://old-service",
        template=stored,
    )
    constructor.assert_called_once_with("http://live-service")
    executable.get_model_template_config_with_digest.assert_called_once_with()
