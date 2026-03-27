"""End-to-end tests for chapkit self-registration flow.

Verifies that services registered via the v2 Orchestrator appear in
GET /v1/crud/model-templates with correct health status and configured models.
"""

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

MOCK_INFO_DICT = {
    "id": "test-model",
    "display_name": "Test Model",
    "version": "1.0.0",
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
