"""End-to-end tests for chapkit self-registration and template registration.

A registered service is a liveness signal for a stored template, not a template
itself: templates are stored through POST /v1/crud/model-templates (what
chap-admin install calls) and reported with a health status on
GET /v1/crud/model-templates.
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
from chap_core.database.model_templates_and_config_tables import ModelTemplateDB
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

# What chap-admin install posts for the same service, as a marketplace entry describes it.
TEMPLATE = {
    "name": "test-model",
    "version": "1.0.0",
    "sourceDigest": "a" * 40,
    "sourceUrl": "http://marketplace-test-model:8000",
    "usesChapkit": True,
    "displayName": "Test Model",
    "supportedPeriodType": "month",
}

SCHEMA = {
    "properties": {
        "n_lags": {"type": "integer", "default": 3},
        "prediction_periods": {"type": "integer", "default": 1},
    }
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
    cls.return_value.get_config_schema.return_value = SCHEMA
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


def _install(client, **overrides):
    response = client.post("/v1/crud/model-templates", json={**TEMPLATE, **overrides})
    assert response.status_code == 200, response.text
    return response.json()


def _test_model(client):
    matching = [t for t in client.get("/v1/crud/model-templates").json() if t["name"] == "test-model"]
    assert len(matching) == 1
    return matching[0]


def test_registered_service_is_not_a_model_until_it_is_installed(client, register_service, mock_wrapper_cls):
    register_service()
    client.post("/v2/services/$register", json={"url": "http://test-service:8080", "info": MOCK_INFO_DICT})

    assert client.get("/v1/crud/model-templates").json() == []
    assert client.get("/v1/crud/configured-models").json() == []
    # Nothing is read from a service that no template needs.
    mock_wrapper_cls.assert_not_called()


def test_installed_template_reports_its_registered_service_as_live(client, register_service):
    installed = _install(client)
    assert installed["sourceDigest"] == "a" * 40
    assert installed["usesChapkit"] is True
    assert _test_model(client)["healthStatus"] is None

    register_service()
    template = _test_model(client)
    assert template["healthStatus"] == "live"
    assert template["id"] == installed["id"]


def test_installing_the_same_version_again_returns_the_stored_row(client):
    first = _install(client)
    again = _install(client, displayName="Another display name")
    assert again["id"] == first["id"]
    assert again["displayName"] == "Test Model"


def test_installing_another_revision_under_a_stored_version_is_refused(client):
    _install(client)
    response = client.post("/v1/crud/model-templates", json={**TEMPLATE, "sourceDigest": "b" * 40})
    assert response.status_code == 409
    assert "write-once" in response.json()["detail"]
    assert _test_model(client)["sourceDigest"] == "a" * 40


def test_a_new_version_is_a_new_live_row_and_the_old_one_stays(client, db_engine):
    first_id = _install(client)["id"]
    template = _install(client, version="1.0.1", sourceDigest="b" * 40)
    assert template["id"] != first_id
    # The stored version stays live until the new one can run, that is, has a configuration.
    assert _test_model(client)["version"] == "1.0.0"
    client.post("/v1/crud/configured-models", json={"name": "default", "modelTemplateId": template["id"]})
    assert _test_model(client)["version"] == "1.0.1"
    with Session(db_engine) as session:
        superseded = session.get(ModelTemplateDB, first_id)
        assert superseded is not None
        assert superseded.is_live is False


@pytest.mark.parametrize("reported", ["b" * 40, None])
def test_republished_service_under_the_same_version_is_a_revision_mismatch(
    client, register_service, mock_wrapper_cls, reported
):
    _install(client)
    register_service({**MOCK_INFO_DICT, "git_revision": reported})
    template = _test_model(client)
    assert template["healthStatus"] == "revision_mismatch"
    assert template["sourceDigest"] == "a" * 40
    assert template["archived"] is False
    # Nothing is fetched from a mismatched service.
    mock_wrapper_cls.assert_not_called()


def test_registration_response_tells_a_service_without_a_git_revision_what_to_do(client):
    payload = {"url": "http://test-service:8080", "info": {**MOCK_INFO_DICT, "git_revision": None}}
    response = client.post("/v2/services/$register", json=payload)

    assert response.status_code == 200
    assert "GIT_REVISION build arg" in response.json()["message"]


def test_registration_response_reports_a_revision_mismatch(client):
    _install(client)
    payload = {"url": "http://test-service:8080", "info": {**MOCK_INFO_DICT, "git_revision": "b" * 40}}
    response = client.post("/v2/services/$register", json=payload)

    assert response.status_code == 200
    message = response.json()["message"]
    assert "version '1.0.0'" in message
    assert "a" * 40 in message and "b" * 40 in message
    assert "info.version" in message


def test_redeploying_the_stored_revision_clears_the_mismatch(client, register_service):
    _install(client)
    register_service({**MOCK_INFO_DICT, "git_revision": "b" * 40})
    assert _test_model(client)["healthStatus"] == "revision_mismatch"

    # The mismatch is computed at read time, so the right image needs no cleanup.
    register_service()
    assert _test_model(client)["healthStatus"] == "live"


def test_user_options_are_filled_from_the_live_service_once(client, register_service, mock_wrapper_cls):
    assert _install(client)["userOptions"] == {}
    register_service()

    assert _test_model(client)["userOptions"] == {"n_lags": {"type": "integer", "default": 3}}
    assert _test_model(client)["userOptions"] == {"n_lags": {"type": "integer", "default": 3}}
    assert mock_wrapper_cls.call_count == 1


def test_schema_fetch_failure_is_retried_on_the_next_listing(client, register_service, mock_wrapper_cls, caplog):
    mock_wrapper_cls.return_value.get_config_schema.side_effect = [
        ConnectionError("service temporarily unavailable"),
        SCHEMA,
    ]
    _install(client)
    register_service()

    with caplog.at_level(logging.WARNING, logger="chap_core.rest_api.v1.routers.crud"):
        template = _test_model(client)
    assert template["userOptions"] == {}
    assert template["healthStatus"] == "live"
    assert any(
        record.levelno == logging.WARNING
        and "Could not fetch config schema" in record.message
        and "will retry next sync" in record.message
        and record.exc_info is not None
        for record in caplog.records
    )
    assert _test_model(client)["userOptions"] == {"n_lags": {"type": "integer", "default": 3}}


def test_deregistered_service_loses_live_status_but_the_template_stays(client, register_service, fake_orchestrator):
    _install(client)
    register_service()
    assert _test_model(client)["healthStatus"] == "live"

    fake_orchestrator.deregister("test-model")

    template = _test_model(client)
    assert template["healthStatus"] is None
    assert template["archived"] is False


def test_retiring_a_template_hides_it_and_its_configured_models_until_it_is_installed_again(client):
    template_id = _install(client)["id"]
    response = client.post(
        "/v1/crud/configured-models",
        json={"name": "tuned", "modelTemplateId": template_id, "userOptionValues": {"n_lags": 5}},
    )
    assert response.status_code == 200
    assert [m["name"] for m in client.get("/v1/crud/configured-models").json()] == ["test-model:tuned"]

    assert client.delete(f"/v1/crud/model-templates/{template_id}").status_code == 200
    assert _test_model(client)["archived"] is True
    assert client.get("/v1/crud/configured-models").json() == []
    assert client.delete("/v1/crud/model-templates/999").status_code == 404

    # Installing again shows the template, and re-adding a configuration shows it too.
    assert _install(client)["archived"] is False
    client.post(
        "/v1/crud/configured-models",
        json={"name": "tuned", "modelTemplateId": template_id, "userOptionValues": {"n_lags": 5}},
    )
    assert [m["name"] for m in client.get("/v1/crud/configured-models").json()] == ["test-model:tuned"]


def test_template_from_a_registered_service(client, register_service):
    register_service()
    response = client.post("/v1/crud/model-templates/from-service", json={"serviceId": "test-model"})
    assert response.status_code == 200, response.text
    template = response.json()
    assert (template["name"], template["version"], template["sourceDigest"]) == ("test-model", "1.0.0", "a" * 40)
    assert template["usesChapkit"] is True
    assert template["userOptions"] == {"n_lags": {"type": "integer", "default": 3}}
    assert _test_model(client)["healthStatus"] == "live"
    # Repeating the call returns the stored row.
    assert (
        client.post("/v1/crud/model-templates/from-service", json={"serviceId": "test-model"}).json()["id"]
        == (template["id"])
    )


def test_template_from_an_unknown_or_unversioned_service_is_refused(client, register_service):
    assert client.post("/v1/crud/model-templates/from-service", json={"serviceId": "test-model"}).status_code == 404
    register_service({**MOCK_INFO_DICT, "git_revision": None})
    response = client.post("/v1/crud/model-templates/from-service", json={"serviceId": "test-model"})
    assert response.status_code == 409
    assert "GIT_REVISION build arg" in response.json()["detail"]
    assert client.get("/v1/crud/model-templates").json() == []


def test_template_from_an_unreachable_service_is_a_bad_gateway(client, register_service, mock_wrapper_cls):
    mock_wrapper_cls.return_value.get_config_schema.side_effect = ConnectionError("service unreachable")
    register_service()
    response = client.post("/v1/crud/model-templates/from-service", json={"serviceId": "test-model"})
    assert response.status_code == 502
    assert client.get("/v1/crud/model-templates").json() == []


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


def test_returns_200_when_redis_unavailable(client):
    with patch(
        "chap_core.rest_api.v2.dependencies.get_orchestrator",
        side_effect=ConnectionError("Redis unavailable"),
    ):
        response = client.get("/v1/crud/model-templates")

    assert response.status_code == 200
    assert response.json() == []
