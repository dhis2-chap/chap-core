import fakeredis
import pytest
from fastapi.testclient import TestClient

from chap_core.rest_api.app import app
from chap_core.rest_api.services.orchestrator import Orchestrator
from chap_core.rest_api.v2.dependencies import get_orchestrator


@pytest.fixture
def fake_redis():
    return fakeredis.FakeRedis()


@pytest.fixture
def test_orchestrator(fake_redis):
    return Orchestrator(redis_client=fake_redis)


@pytest.fixture
def client(test_orchestrator):
    from chap_core.rest_api.v2.rest_api import app as v2_app

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
            "info": {"name": "test-model"},
        }

        response = client.post("/v2/services/$register", json=payload)

        assert response.status_code == 200
        assert response.json()["status"] == "registered"
