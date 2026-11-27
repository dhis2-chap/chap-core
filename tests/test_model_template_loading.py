from types import SimpleNamespace
from unittest.mock import Mock
from typing import cast

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
def test_get_model_template_with_code_non_chapkit(monkeypatch, name, source_url, loaded_name, ignore_env, should_fail):
    wrapper = db.SessionWrapper()
    stored = ModelTemplateDB(name=name, version="v1", source_url=source_url)
    monkeypatch.setattr(wrapper, "get_model_template", lambda _: stored)
    executable = cast(
        db.ModelTemplate,
        SimpleNamespace(model_template_config=SimpleNamespace(name=loaded_name, version="v1")),
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
    from chap_core.rest_api.services import orchestrator as orchestrator_module
    from chap_core.rest_api.v2 import dependencies as v2_dependencies

    # Inject a mocked database session
    session = Mock()
    wrapper = db.SessionWrapper(session=session)
    stored = ModelTemplateDB(
        name="service",
        version="v1",
        source_url="http://old-service",
        uses_chapkit=True,
        source_digest="abc123",
    )
    monkeypatch.setattr(wrapper, "get_model_template", lambda _: stored)
    # Mock Redis dependency
    redis_client = Mock()
    get_redis = Mock(return_value=redis_client)
    monkeypatch.setattr(v2_dependencies, "get_redis", get_redis)
    # Mock the service registry
    orchestrator = Mock()
    orchestrator.get.return_value = SimpleNamespace(url="http://live-service")
    orchestrator_constructor = Mock(return_value=orchestrator)
    monkeypatch.setattr(
        orchestrator_module,
        "Orchestrator",
        orchestrator_constructor,
    )
    # Mock the Chapkit executable
    executable = Mock()
    executable.get_model_template_config_with_digest.return_value = (
        SimpleNamespace(
            name=reported_name,
            version=reported_version,
        ),
        reported_digest,
    )
    constructor = Mock(return_value=executable)
    monkeypatch.setattr(
        db,
        "ExternalChapkitModelTemplate",
        constructor,
    )
    # Execute and verify result
    if expected_error:
        with pytest.raises(expected_error):
            wrapper.get_model_template_with_code(1)
        executable.close.assert_called_once_with()
    else:
        assert wrapper.get_model_template_with_code(1) is executable
        executable.close.assert_not_called()
    # Verify the live service was resolved through the registry
    get_redis.assert_called_once_with()
    orchestrator_constructor.assert_called_once_with(redis_client=redis_client)
    orchestrator.get.assert_called_once_with("service")
    # Verify stale source URL was updated and committed
    assert stored.source_url == "http://live-service"
    session.commit.assert_called_once_with()
    # Verify the executable was loaded from the live URL
    constructor.assert_called_once_with("http://live-service")
    executable.get_model_template_config_with_digest.assert_called_once_with()
