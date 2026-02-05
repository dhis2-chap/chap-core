import fakeredis
import pytest
from fastapi.testclient import TestClient

from chap_core.rest_api.app import app
from chap_core.rest_api.services.orchestrator import Orchestrator
from chap_core.rest_api.v2.dependencies import SERVICE_KEY_ENV_VAR, get_orchestrator

TEST_SERVICE_KEY = "test-service-key"


@pytest.fixture
def fake_redis():
    return fakeredis.FakeRedis()


@pytest.fixture
def test_orchestrator(fake_redis):
    return Orchestrator(redis_client=fake_redis)


@pytest.fixture
def client(test_orchestrator, monkeypatch):
    from chap_core.rest_api.v2.rest_api import app as v2_app

    monkeypatch.setenv(SERVICE_KEY_ENV_VAR, TEST_SERVICE_KEY)

    def override_get_orchestrator():
        return test_orchestrator

    v2_app.dependency_overrides[get_orchestrator] = override_get_orchestrator
    yield TestClient(app, raise_server_exceptions=False)
    v2_app.dependency_overrides.clear()


class TestParentApp:
    def test_root_health_endpoint(self, client):
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_v1_health_endpoint(self, client):
        response = client.get("/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

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

    def test_docs_redirects_to_v1(self, client):
        response = client.get("/docs", follow_redirects=False)

        assert response.status_code == 307
        assert response.headers["location"] == "/v1/docs"

    def test_redoc_redirects_to_v1(self, client):
        response = client.get("/redoc", follow_redirects=False)

        assert response.status_code == 307
        assert response.headers["location"] == "/v1/redoc"

    def test_openapi_json_redirects_to_v1(self, client):
        response = client.get("/openapi.json", follow_redirects=False)

        assert response.status_code == 307
        assert response.headers["location"] == "/v1/openapi.json"

    def test_v1_docs_accessible(self, client):
        response = client.get("/v1/docs")

        assert response.status_code == 200

    def test_v2_docs_accessible(self, client):
        response = client.get("/v2/docs")

        assert response.status_code == 200
