"""End-to-end tests for chapkit self-registration flow.

Verifies that services registered via the v2 Orchestrator appear in
GET /v1/crud/model-templates with correct health status and configured models.
"""

import logging
from unittest.mock import MagicMock, patch

import fakeredis
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel

import chap_core.database.tables  # noqa: F401 - ensure all table models are registered with SQLModel
from chap_core.rest_api.app import app
from chap_core.rest_api.services.orchestrator import Orchestrator
from chap_core.rest_api.services.schemas import MLServiceInfo, RegistrationRequest
from chap_core.rest_api.v1.routers.dependencies import get_session
from chap_core.rest_api.v2.dependencies import get_orchestrator

MOCK_INFO_DICT = {
    "id": "test-model",
    "display_name": "Test Model",
    "version": "1.0.0",
    "git_revision": "a" * 40,
    "description": "A test model",
    "model_metadata": {
        "author": "Test",
        "author_assessed_status": "yellow",
    },
    "period_type": "monthly",
    "min_prediction_periods": 1,
    "max_prediction_periods": 12,
    "allow_free_additional_continuous_covariates": False,
    "required_covariates": [],
    "requires_geo": False,
}


@pytest.fixture
def fake_orchestrator():
    return Orchestrator(redis_client=fakeredis.FakeRedis())


@pytest.fixture
def db_engine():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def mock_wrapper_cls():
    cls = MagicMock()
    cls.return_value.list_configs.return_value = []
    return cls


@pytest.fixture
def client(db_engine, fake_orchestrator, mock_wrapper_cls):
    def get_test_session():
        with Session(db_engine) as session:
            yield session

    app.dependency_overrides[get_session] = get_test_session
    # The v2 register endpoint takes the orchestrator through Depends, while the lazy
    # v1 template sync calls the factory directly, so both need the fake.
    app.dependency_overrides[get_orchestrator] = lambda: fake_orchestrator

    with (
        patch("chap_core.rest_api.v2.dependencies.get_orchestrator", return_value=fake_orchestrator),
        patch("chap_core.models.chapkit_rest_api_wrapper.CHAPKitRestAPIWrapper", mock_wrapper_cls),
    ):
        yield TestClient(app, raise_server_exceptions=False)

    app.dependency_overrides.clear()


@pytest.fixture
def register_service(fake_orchestrator):
    def _register(info_dict=None):
        info_dict = info_dict or MOCK_INFO_DICT
        info = MLServiceInfo.model_validate(info_dict)
        request = RegistrationRequest(url="http://test-service:8080", info=info)
        fake_orchestrator.register(request)

    return _register


def test_registered_service_appears_in_model_templates(client, register_service):
    register_service()

    response = client.get("/v1/crud/model-templates")

    assert response.status_code == 200
    templates = response.json()
    matching = [t for t in templates if t["name"] == "test-model"]
    assert len(matching) == 1
    assert matching[0]["healthStatus"] == "live"


def test_registered_service_git_revision_is_stored_as_source_digest(client, register_service):
    register_service({**MOCK_INFO_DICT, "git_revision": "b" * 40})

    templates = client.get("/v1/crud/model-templates").json()
    matching = [t for t in templates if t["name"] == "test-model"]
    assert len(matching) == 1
    assert matching[0]["sourceDigest"] == "b" * 40


def _test_model(client):
    matching = [t for t in client.get("/v1/crud/model-templates").json() if t["name"] == "test-model"]
    assert len(matching) == 1
    return matching[0]


@pytest.mark.parametrize("stored, reported", [("a" * 40, "b" * 40), ("a" * 40, None)])
def test_republished_service_under_the_same_version_is_a_revision_mismatch(
    client, register_service, mock_wrapper_cls, stored, reported
):
    register_service({**MOCK_INFO_DICT, "git_revision": stored})
    assert _test_model(client)["sourceDigest"] == stored
    service_calls = mock_wrapper_cls.call_count

    # Republishing the same version from another commit leaves the stored row alone.
    register_service({**MOCK_INFO_DICT, "git_revision": reported})
    template = _test_model(client)
    assert template["healthStatus"] == "revision_mismatch"
    assert template["sourceDigest"] == stored
    assert template["archived"] is False
    # Nothing is fetched from a mismatched service, so its configured models are not re-synced.
    assert mock_wrapper_cls.call_count == service_calls


@pytest.mark.parametrize("git_revision", [None, ""])
def test_service_without_a_git_revision_is_not_stored_until_it_reports_one(client, register_service, git_revision):
    """Storing a row without a digest would burn the version label, so the label stays free."""
    register_service({**MOCK_INFO_DICT, "git_revision": git_revision})
    assert [t for t in client.get("/v1/crud/model-templates").json() if t["name"] == "test-model"] == []
    assert client.get("/v1/crud/configured-models").json() == []

    # The same version, rebuilt with the build arg, is stored and live.
    register_service({**MOCK_INFO_DICT, "git_revision": "a" * 40})
    template = _test_model(client)
    assert template["version"] == "1.0.0"
    assert template["sourceDigest"] == "a" * 40
    assert template["healthStatus"] == "live"


def test_registration_response_tells_a_service_without_a_git_revision_what_to_do(client):
    payload = {"url": "http://test-service:8080", "info": {**MOCK_INFO_DICT, "git_revision": None}}
    response = client.post("/v2/services/$register", json=payload)

    assert response.status_code == 200
    assert "GIT_REVISION build arg" in response.json()["message"]


def test_redeploying_the_stored_revision_clears_the_mismatch(client, register_service):
    register_service()
    assert _test_model(client)["healthStatus"] == "live"
    register_service({**MOCK_INFO_DICT, "git_revision": "b" * 40})
    assert _test_model(client)["healthStatus"] == "revision_mismatch"

    # The mismatch is computed at read time, so the right image needs no cleanup.
    register_service()
    assert _test_model(client)["healthStatus"] == "live"


def test_version_bump_after_a_revision_mismatch_creates_a_new_live_row(client, register_service):
    register_service()
    first_id = _test_model(client)["id"]
    register_service({**MOCK_INFO_DICT, "git_revision": "b" * 40})
    assert _test_model(client)["healthStatus"] == "revision_mismatch"

    register_service({**MOCK_INFO_DICT, "git_revision": "b" * 40, "version": "1.0.1"})
    template = _test_model(client)
    assert template["id"] != first_id
    assert template["version"] == "1.0.1"
    assert template["sourceDigest"] == "b" * 40
    assert template["healthStatus"] == "live"


def test_registration_response_reports_a_revision_mismatch(client, register_service):
    register_service()
    client.get("/v1/crud/model-templates")

    payload = {"url": "http://test-service:8080", "info": {**MOCK_INFO_DICT, "git_revision": "b" * 40}}
    response = client.post("/v2/services/$register", json=payload)

    assert response.status_code == 200
    message = response.json()["message"]
    assert "version '1.0.0'" in message
    assert "a" * 40 in message and "b" * 40 in message
    assert "info.version" in message


def test_schema_fetch_failure_does_not_freeze_empty_user_options(client, register_service, mock_wrapper_cls, caplog):
    schema = {
        "properties": {
            "n_lags": {"type": "integer", "default": 3},
            "prediction_periods": {"type": "integer", "default": 1},
        }
    }
    mock_wrapper_cls.return_value.get_config_schema.side_effect = [
        ConnectionError("service temporarily unavailable"),
        schema,
    ]
    register_service()

    # An incomplete template is not created while the schema endpoint is down.
    with caplog.at_level(logging.WARNING, logger="chap_core.rest_api.v1.routers.crud"):
        assert client.get("/v1/crud/model-templates").json() == []
    assert any(
        record.levelno == logging.WARNING
        and "Could not fetch config schema" in record.message
        and "will retry next sync" in record.message
        and record.exc_info is not None
        for record in caplog.records
    )

    # The next sync succeeds and creates the template with its user options.
    templates = client.get("/v1/crud/model-templates").json()
    matching = [template for template in templates if template["name"] == "test-model"]
    assert len(matching) == 1
    assert matching[0]["userOptions"] == {"n_lags": {"type": "integer", "default": 3}}


def test_registered_service_has_configured_model(client, register_service):
    register_service()
    # Trigger sync
    client.get("/v1/crud/model-templates")

    response = client.get("/v1/crud/configured-models")

    assert response.status_code == 200
    models = response.json()
    # Default configuration uses template name as configured model name
    chapkit_models = [m for m in models if m["name"] == "test-model"]
    assert len(chapkit_models) == 1
    assert chapkit_models[0]["usesChapkit"] is True


def test_creates_default_config_when_no_configs(client, register_service, mock_wrapper_cls):
    mock_wrapper_cls.return_value.list_configs.return_value = []
    register_service()

    client.get("/v1/crud/model-templates")

    response = client.get("/v1/crud/configured-models")
    models = response.json()
    assert len(models) == 1
    # Default configuration uses template name as configured model name
    assert models[0]["name"] == "test-model"


def test_non_chapkit_template_has_null_health_status(client, db_engine):
    from chap_core.database.database import SessionWrapper
    from chap_core.models.external_chapkit_model import ml_service_info_to_model_template_config

    info = MLServiceInfo.model_validate({**MOCK_INFO_DICT, "id": "non-chapkit-model"})
    config = ml_service_info_to_model_template_config(info, "http://localhost:9999")

    with Session(db_engine) as session:
        wrapper = SessionWrapper(session=session)
        wrapper.add_model_template_from_yaml_config(config)

    response = client.get("/v1/crud/model-templates")

    templates = response.json()
    matching = [t for t in templates if t["name"] == "non-chapkit-model"]
    assert len(matching) == 1
    assert matching[0]["healthStatus"] is None


def test_deregistered_service_loses_live_status(client, register_service, fake_orchestrator):
    register_service()
    response = client.get("/v1/crud/model-templates")
    templates = response.json()
    assert any(t["name"] == "test-model" and t["healthStatus"] == "live" for t in templates)

    fake_orchestrator.deregister("test-model")

    response = client.get("/v1/crud/model-templates")
    templates = response.json()
    matching = [t for t in templates if t["name"] == "test-model"]
    assert len(matching) == 1
    assert matching[0]["healthStatus"] is None


def test_sync_is_idempotent(client, register_service):
    register_service()

    response1 = client.get("/v1/crud/model-templates")
    response2 = client.get("/v1/crud/model-templates")

    templates1 = response1.json()
    templates2 = response2.json()
    assert len(templates1) == len(templates2)

    models1 = client.get("/v1/crud/configured-models").json()
    models2 = client.get("/v1/crud/configured-models").json()
    assert len(models1) == len(models2)


def test_deregistered_service_becomes_archived(client, register_service, fake_orchestrator):
    register_service()
    response = client.get("/v1/crud/model-templates")
    templates = response.json()
    matching = [t for t in templates if t["name"] == "test-model"]
    assert len(matching) == 1
    assert matching[0]["archived"] is False

    fake_orchestrator.deregister("test-model")

    response = client.get("/v1/crud/model-templates")
    templates = response.json()
    matching = [t for t in templates if t["name"] == "test-model"]
    assert len(matching) == 1
    assert matching[0]["archived"] is True


def test_re_registered_service_becomes_unarchived(client, register_service, fake_orchestrator):
    register_service()
    client.get("/v1/crud/model-templates")

    fake_orchestrator.deregister("test-model")
    response = client.get("/v1/crud/model-templates")
    matching = [t for t in response.json() if t["name"] == "test-model"]
    assert matching[0]["archived"] is True

    register_service()
    response = client.get("/v1/crud/model-templates")
    matching = [t for t in response.json() if t["name"] == "test-model"]
    assert len(matching) == 1
    assert matching[0]["archived"] is False
    assert matching[0]["healthStatus"] == "live"


def test_returns_200_when_redis_unavailable(client):
    with patch(
        "chap_core.rest_api.v2.dependencies.get_orchestrator",
        side_effect=ConnectionError("Redis unavailable"),
    ):
        response = client.get("/v1/crud/model-templates")

    assert response.status_code == 200
    assert response.json() == []


def test_template_archived_when_config_sync_never_succeeded(
    client, register_service, fake_orchestrator, mock_wrapper_cls
):
    """A template whose initial config sync failed should still be archived when the service deregisters."""
    # Config fetch fails on every attempt
    mock_wrapper_cls.return_value.list_configs.side_effect = ConnectionError("service unreachable")

    register_service()
    client.get("/v1/crud/model-templates")

    # Verify template exists but has no configured models
    templates = client.get("/v1/crud/model-templates").json()
    matching = [t for t in templates if t["name"] == "test-model"]
    assert len(matching) == 1

    configured = client.get("/v1/crud/configured-models").json()
    chapkit_configured = [m for m in configured if m["name"] == "test-model"]
    assert len(chapkit_configured) == 0

    # Deregister the service
    fake_orchestrator.deregister("test-model")

    # Template should be archived even though it has no configured models
    templates = client.get("/v1/crud/model-templates").json()
    matching = [t for t in templates if t["name"] == "test-model"]
    assert len(matching) == 1
    assert matching[0]["archived"] is True


def test_creates_multiple_configured_models_from_service_configs(client, register_service, mock_wrapper_cls):
    """When a chapkit service exposes multiple named configs, each becomes a configured model."""
    config_a = MagicMock()
    config_a.name = "config-a"
    config_b = MagicMock()
    config_b.name = "config-b"
    mock_wrapper_cls.return_value.list_configs.return_value = [config_a, config_b]

    register_service()
    client.get("/v1/crud/model-templates")

    models = client.get("/v1/crud/configured-models").json()
    names = {m["name"] for m in models}
    assert "test-model:config-a" in names
    assert "test-model:config-b" in names
    assert len(names) == 2


def test_re_registered_service_with_new_version_adds_new_template(
    client, register_service, fake_orchestrator, db_engine
):
    """A service with a new version adds a template version. It does not change the old one."""
    register_service()
    templates = client.get("/v1/crud/model-templates").json()
    matching = [t for t in templates if t["name"] == "test-model"]
    assert len(matching) == 1
    assert matching[0]["version"] == "1.0.0"
    assert matching[0]["requiresGeo"] is False
    first_id = matching[0]["id"]

    # Re-register with updated metadata
    updated_info = {
        **MOCK_INFO_DICT,
        "version": "2.0.0",
        "display_name": "Updated Model",
        "requires_geo": True,
        "required_covariates": ["rainfall"],
    }
    fake_orchestrator.deregister("test-model")
    register_service(info_dict=updated_info)

    # Only the live version is listed.
    templates = client.get("/v1/crud/model-templates").json()
    matching = [t for t in templates if t["name"] == "test-model"]
    assert len(matching) == 1
    live = matching[0]
    assert live["isLive"] is True
    assert live["version"] == "2.0.0"
    assert live["displayName"] == "Updated Model"
    assert live["requiresGeo"] is True
    assert live["requiredCovariates"] == ["rainfall"]
    assert live["id"] != first_id
    # The old version keeps its row and id, but is no longer live.
    from chap_core.database.model_templates_and_config_tables import ModelTemplateDB

    with Session(db_engine) as session:
        superseded = session.get(ModelTemplateDB, first_id)
        assert superseded is not None
        assert superseded.version == "1.0.0"
        assert superseded.is_live is False


def test_non_chapkit_orphan_template_not_archived(client, register_service, fake_orchestrator, db_engine):
    """A non-chapkit template with zero configured models must not be archived by chapkit sync."""
    from chap_core.database.model_templates_and_config_tables import ModelTemplateDB

    # Create a non-chapkit template with no configured models
    with Session(db_engine) as session:
        template = ModelTemplateDB(name="manual-template", version="1.0.0", source_url="http://example.com")
        session.add(template)
        session.commit()

    # Register and deregister a chapkit service (triggers archival)
    register_service()
    client.get("/v1/crud/model-templates")
    fake_orchestrator.deregister("test-model")
    client.get("/v1/crud/model-templates")

    # Non-chapkit template must NOT be archived
    templates = client.get("/v1/crud/model-templates").json()
    manual = [t for t in templates if t["name"] == "manual-template"]
    assert len(manual) == 1
    assert manual[0]["archived"] is False

    # Chapkit template should be archived
    chapkit = [t for t in templates if t["name"] == "test-model"]
    assert len(chapkit) == 1
    assert chapkit[0]["archived"] is True


def test_config_sync_frozen_after_first_discovery(client, register_service, mock_wrapper_cls):
    """New configs added to a service after first discovery are not synced (intentional)."""
    config_a = MagicMock()
    config_a.name = "config-a"
    mock_wrapper_cls.return_value.list_configs.return_value = [config_a]

    register_service()
    client.get("/v1/crud/model-templates")

    # Service now exposes a second config
    config_b = MagicMock()
    config_b.name = "config-b"
    mock_wrapper_cls.return_value.list_configs.return_value = [config_a, config_b]

    # Trigger another sync
    client.get("/v1/crud/model-templates")

    models = client.get("/v1/crud/configured-models").json()
    names = {m["name"] for m in models}
    assert "test-model:config-a" in names
    assert "test-model:config-b" not in names  # intentionally not synced
