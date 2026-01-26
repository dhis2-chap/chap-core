import fakeredis
import pytest
from fastapi.testclient import TestClient

from chap_core.rest_api.services.orchestrator import Orchestrator
from chap_core.rest_api.v2.dependencies import (
    SERVICE_KEY_ENV_VAR,
    get_orchestrator,
)
from chap_core.rest_api.v2.rest_api import app

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

    def override_get_orchestrator():
        return test_orchestrator

    app.dependency_overrides[get_orchestrator] = override_get_orchestrator
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers():
    return {"X-Service-Key": TEST_SERVICE_KEY}


@pytest.fixture
def sample_registration():
    return {
        "url": "http://model-service:8080",
        "info": {
            "id": "test-model",
            "display_name": "Test Model",
            "model_metadata": {"author": "Test Author"},
            "period_type": "monthly",
        },
    }


class TestRegisterEndpoint:
    def test_register_service_returns_200(self, client, sample_registration, auth_headers):
        response = client.post("/services/$register", json=sample_registration, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "registered"
        assert data["service_url"] == sample_registration["url"]
        assert "id" in data
        assert "ping_url" in data

    def test_register_service_invalid_payload(self, client, auth_headers):
        response = client.post("/services/$register", json={}, headers=auth_headers)

        assert response.status_code == 422

    def test_register_service_invalid_slug_uppercase(self, client, auth_headers):
        payload = {
            "url": "http://model-service:8080",
            "info": {
                "id": "Invalid-Slug",
                "display_name": "Test",
                "model_metadata": {"author": "Test"},
                "period_type": "monthly",
            },
        }
        response = client.post("/services/$register", json=payload, headers=auth_headers)

        assert response.status_code == 422

    def test_register_service_invalid_slug_starts_with_number(self, client, auth_headers):
        payload = {
            "url": "http://model-service:8080",
            "info": {
                "id": "123-service",
                "display_name": "Test",
                "model_metadata": {"author": "Test"},
                "period_type": "monthly",
            },
        }
        response = client.post("/services/$register", json=payload, headers=auth_headers)

        assert response.status_code == 422


class TestPingEndpoint:
    def test_ping_registered_service(self, client, sample_registration, auth_headers):
        reg_response = client.post("/services/$register", json=sample_registration, headers=auth_headers)
        service_id = reg_response.json()["id"]

        response = client.put(f"/services/{service_id}/$ping", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert data["id"] == service_id

    def test_ping_nonexistent_service(self, client, auth_headers):
        response = client.put("/services/nonexistent-id/$ping", headers=auth_headers)

        assert response.status_code == 404


class TestListEndpoint:
    def test_list_empty_services(self, client):
        response = client.get("/services")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["services"] == []

    def test_list_registered_services(self, client, auth_headers):
        reg1 = {
            "url": "http://service-one:8080",
            "info": {
                "id": "service-one",
                "display_name": "Service One",
                "model_metadata": {"author": "Author"},
                "period_type": "monthly",
            },
        }
        reg2 = {
            "url": "http://service-two:8080",
            "info": {
                "id": "service-two",
                "display_name": "Service Two",
                "model_metadata": {"author": "Author"},
                "period_type": "monthly",
            },
        }
        client.post("/services/$register", json=reg1, headers=auth_headers)
        client.post("/services/$register", json=reg2, headers=auth_headers)

        response = client.get("/services")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["services"]) == 2


class TestGetEndpoint:
    def test_get_registered_service(self, client, sample_registration, auth_headers):
        reg_response = client.post("/services/$register", json=sample_registration, headers=auth_headers)
        service_id = reg_response.json()["id"]

        response = client.get(f"/services/{service_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == service_id
        assert data["url"] == sample_registration["url"]
        assert data["info"]["display_name"] == sample_registration["info"]["display_name"]
        assert data["info"]["period_type"] == sample_registration["info"]["period_type"]

    def test_get_nonexistent_service(self, client):
        response = client.get("/services/nonexistent-id")

        assert response.status_code == 404


class TestDeregisterEndpoint:
    def test_deregister_service(self, client, sample_registration, auth_headers):
        reg_response = client.post("/services/$register", json=sample_registration, headers=auth_headers)
        service_id = reg_response.json()["id"]

        response = client.delete(f"/services/{service_id}", headers=auth_headers)

        assert response.status_code == 204

        # Verify service is gone
        get_response = client.get(f"/services/{service_id}")
        assert get_response.status_code == 404

    def test_deregister_nonexistent_service(self, client, auth_headers):
        response = client.delete("/services/nonexistent-id", headers=auth_headers)

        assert response.status_code == 404


class TestFullRegistrationFlow:
    def test_complete_lifecycle(self, client, sample_registration, auth_headers):
        # Register
        reg_response = client.post("/services/$register", json=sample_registration, headers=auth_headers)
        assert reg_response.status_code == 200
        service_id = reg_response.json()["id"]

        # Verify in list
        list_response = client.get("/services")
        assert list_response.json()["count"] == 1

        # Ping to keep alive
        ping_response = client.put(f"/services/{service_id}/$ping", headers=auth_headers)
        assert ping_response.status_code == 200

        # Get details
        get_response = client.get(f"/services/{service_id}")
        assert get_response.status_code == 200
        assert get_response.json()["url"] == sample_registration["url"]

        # Deregister
        del_response = client.delete(f"/services/{service_id}", headers=auth_headers)
        assert del_response.status_code == 204

        # Verify removed
        final_list = client.get("/services")
        assert final_list.json()["count"] == 0


class TestServiceKeyAuthentication:
    @pytest.fixture
    def client_without_env_var(self, test_orchestrator, monkeypatch):
        monkeypatch.delenv(SERVICE_KEY_ENV_VAR, raising=False)

        def override_get_orchestrator():
            return test_orchestrator

        app.dependency_overrides[get_orchestrator] = override_get_orchestrator
        yield TestClient(app, raise_server_exceptions=False)
        app.dependency_overrides.clear()

    def test_register_without_key_returns_422(self, client, sample_registration):
        response = client.post("/services/$register", json=sample_registration)

        assert response.status_code == 422

    def test_register_with_invalid_key_returns_401(self, client, sample_registration):
        response = client.post(
            "/services/$register",
            json=sample_registration,
            headers={"X-Service-Key": "wrong-key"},
        )

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid service key"

    def test_register_with_valid_key_succeeds(self, client, sample_registration, auth_headers):
        response = client.post("/services/$register", json=sample_registration, headers=auth_headers)

        assert response.status_code == 200

    def test_register_without_env_var_allows_registration(self, client_without_env_var, sample_registration):
        response = client_without_env_var.post(
            "/services/$register",
            json=sample_registration,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "registered"

    def test_ping_without_key_returns_422(self, client, sample_registration, auth_headers):
        reg_response = client.post("/services/$register", json=sample_registration, headers=auth_headers)
        service_id = reg_response.json()["id"]

        response = client.put(f"/services/{service_id}/$ping")

        assert response.status_code == 422

    def test_ping_with_invalid_key_returns_401(self, client, sample_registration, auth_headers):
        reg_response = client.post("/services/$register", json=sample_registration, headers=auth_headers)
        service_id = reg_response.json()["id"]

        response = client.put(f"/services/{service_id}/$ping", headers={"X-Service-Key": "wrong-key"})

        assert response.status_code == 401

    def test_deregister_without_key_returns_422(self, client, sample_registration, auth_headers):
        reg_response = client.post("/services/$register", json=sample_registration, headers=auth_headers)
        service_id = reg_response.json()["id"]

        response = client.delete(f"/services/{service_id}")

        assert response.status_code == 422

    def test_deregister_with_invalid_key_returns_401(self, client, sample_registration, auth_headers):
        reg_response = client.post("/services/$register", json=sample_registration, headers=auth_headers)
        service_id = reg_response.json()["id"]

        response = client.delete(f"/services/{service_id}", headers={"X-Service-Key": "wrong-key"})

        assert response.status_code == 401

    def test_list_does_not_require_auth(self, client):
        response = client.get("/services")

        assert response.status_code == 200

    def test_get_does_not_require_auth(self, client, sample_registration, auth_headers):
        reg_response = client.post("/services/$register", json=sample_registration, headers=auth_headers)
        service_id = reg_response.json()["id"]

        response = client.get(f"/services/{service_id}")

        assert response.status_code == 200
