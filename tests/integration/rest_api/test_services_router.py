import fakeredis
import pytest
from fastapi.testclient import TestClient

from chap_core.rest_api.services.orchestrator import Orchestrator
from chap_core.rest_api.v2.dependencies import get_orchestrator
from chap_core.rest_api.v2.rest_api import app


@pytest.fixture
def fake_redis():
    return fakeredis.FakeRedis()


@pytest.fixture
def test_orchestrator(fake_redis):
    return Orchestrator(redis_client=fake_redis)


@pytest.fixture
def client(test_orchestrator):
    def override_get_orchestrator():
        return test_orchestrator

    app.dependency_overrides[get_orchestrator] = override_get_orchestrator
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


@pytest.fixture
def sample_registration():
    return {
        "url": "http://model-service:8080",
        "info": {"name": "test-model", "version": "1.0.0"},
    }


class TestRegisterEndpoint:
    def test_register_service_returns_201(self, client, sample_registration):
        response = client.post("/services/$register", json=sample_registration)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "registered"
        assert data["service_url"] == sample_registration["url"]
        assert "id" in data
        assert "ping_url" in data

    def test_register_service_invalid_payload(self, client):
        response = client.post("/services/$register", json={})

        assert response.status_code == 422


class TestPingEndpoint:
    def test_ping_registered_service(self, client, sample_registration):
        reg_response = client.post("/services/$register", json=sample_registration)
        service_id = reg_response.json()["id"]

        response = client.put(f"/services/{service_id}/$ping")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert data["id"] == service_id

    def test_ping_nonexistent_service(self, client):
        response = client.put("/services/nonexistent-id/$ping")

        assert response.status_code == 404


class TestListEndpoint:
    def test_list_empty_services(self, client):
        response = client.get("/services")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["services"] == []

    def test_list_registered_services(self, client, sample_registration):
        client.post("/services/$register", json=sample_registration)
        client.post("/services/$register", json=sample_registration)

        response = client.get("/services")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["services"]) == 2


class TestGetEndpoint:
    def test_get_registered_service(self, client, sample_registration):
        reg_response = client.post("/services/$register", json=sample_registration)
        service_id = reg_response.json()["id"]

        response = client.get(f"/services/{service_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == service_id
        assert data["url"] == sample_registration["url"]
        assert data["info"] == sample_registration["info"]

    def test_get_nonexistent_service(self, client):
        response = client.get("/services/nonexistent-id")

        assert response.status_code == 404


class TestDeregisterEndpoint:
    def test_deregister_service(self, client, sample_registration):
        reg_response = client.post("/services/$register", json=sample_registration)
        service_id = reg_response.json()["id"]

        response = client.delete(f"/services/{service_id}")

        assert response.status_code == 204

        # Verify service is gone
        get_response = client.get(f"/services/{service_id}")
        assert get_response.status_code == 404

    def test_deregister_nonexistent_service(self, client):
        response = client.delete("/services/nonexistent-id")

        assert response.status_code == 404


class TestFullRegistrationFlow:
    def test_complete_lifecycle(self, client, sample_registration):
        # Register
        reg_response = client.post("/services/$register", json=sample_registration)
        assert reg_response.status_code == 200
        service_id = reg_response.json()["id"]

        # Verify in list
        list_response = client.get("/services")
        assert list_response.json()["count"] == 1

        # Ping to keep alive
        ping_response = client.put(f"/services/{service_id}/$ping")
        assert ping_response.status_code == 200

        # Get details
        get_response = client.get(f"/services/{service_id}")
        assert get_response.status_code == 200
        assert get_response.json()["url"] == sample_registration["url"]

        # Deregister
        del_response = client.delete(f"/services/{service_id}")
        assert del_response.status_code == 204

        # Verify removed
        final_list = client.get("/services")
        assert final_list.json()["count"] == 0
