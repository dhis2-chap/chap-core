import fakeredis
import pytest
from fastapi.testclient import TestClient

from chap_core.rest_api.app import app
from chap_core.rest_api.services.orchestrator import Orchestrator
from chap_core.rest_api.v1.routers.dependencies import get_settings
from chap_core.rest_api.v2.dependencies import SERVICE_KEY_ENV_VAR, get_orchestrator
from chap_core.rest_api.worker_functions import WorkerConfig

TEST_SERVICE_KEY = "test-service-key"


@pytest.fixture
def fake_redis():
    return fakeredis.FakeRedis()


@pytest.fixture
def test_orchestrator(fake_redis):
    return Orchestrator(redis_client=fake_redis)


@pytest.fixture
def client(test_orchestrator, monkeypatch):
    monkeypatch.setenv(SERVICE_KEY_ENV_VAR, TEST_SERVICE_KEY)

    app.dependency_overrides[get_orchestrator] = lambda: test_orchestrator
    app.dependency_overrides[get_settings] = lambda: WorkerConfig()
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


class TestCommonEndpoints:
    def test_root_health_endpoint(self, client):
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "success", "message": "healthy"}

    def test_root_version_endpoint(self, client):
        response = client.get("/version")

        assert response.status_code == 200
        data = response.json()
        assert "version" in data

    def test_root_status_endpoint(self, client):
        response = client.get("/status")

        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True
        assert data["status"] == "idle"

    def test_root_is_compatible_endpoint(self, client):
        response = client.get("/is-compatible?modelling_app_version=999.0.0")

        assert response.status_code == 200
        data = response.json()
        assert data["compatible"] is True

    def test_root_system_info_endpoint(self, client):
        response = client.get("/system-info")

        assert response.status_code == 200
        data = response.json()
        assert "chap_core_version" in data
        assert "python_version" in data
        assert "os" in data

    def test_root_get_exception_endpoint(self, client):
        response = client.get("/get-exception")

        assert response.status_code == 200
        assert response.json() == ""

    def test_root_cancel_endpoint(self, client):
        response = client.post("/cancel")

        assert response.status_code == 200
        assert response.json() == {"status": "success"}


class TestV2Mounting:
    def test_v2_services_endpoint(self, client):
        response = client.get("/v2/services")

        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "services" in data

    def test_v2_register_endpoint(self, client):
        payload = {
            "url": "http://model-service:8080",
            "info": {
                "id": "test-model",
                "display_name": "Test Model",
                "model_metadata": {"author": "Test Author"},
                "period_type": "monthly",
            },
        }

        response = client.post(
            "/v2/services/$register",
            json=payload,
            headers={"X-Service-Key": TEST_SERVICE_KEY},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "registered"


class TestDocs:
    def test_docs_accessible(self, client):
        response = client.get("/docs")

        assert response.status_code == 200

    def test_redoc_accessible(self, client):
        response = client.get("/redoc")

        assert response.status_code == 200
