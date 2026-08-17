"""Tests for the opt-in CHAP_API_TOKEN gate."""

import fakeredis
import pytest
from fastapi.testclient import TestClient

from chap_core.rest_api.app import app
from chap_core.rest_api.auth import (
    API_TOKEN_ENV_VAR,
    MIN_TOKEN_LENGTH,
    SERVICE_KEY_ENV_VAR,
    warn_on_weak_token,
)
from chap_core.rest_api.services.orchestrator import Orchestrator
from chap_core.rest_api.v1.routers.dependencies import get_settings
from chap_core.rest_api.v2.dependencies import get_orchestrator
from chap_core.rest_api.worker_functions import WorkerConfig

TEST_TOKEN = "test-api-token"
TEST_SERVICE_KEY = "test-service-key"

# Any gated route works here; this one needs no database.
GATED_PATH = "/v2/services"


@pytest.fixture
def test_orchestrator():
    return Orchestrator(redis_client=fakeredis.FakeRedis())


@pytest.fixture
def unauthenticated_client(test_orchestrator, monkeypatch):
    monkeypatch.delenv(API_TOKEN_ENV_VAR, raising=False)
    app.dependency_overrides[get_orchestrator] = lambda: test_orchestrator
    app.dependency_overrides[get_settings] = lambda: WorkerConfig()
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


@pytest.fixture
def client(test_orchestrator, monkeypatch):
    monkeypatch.setenv(API_TOKEN_ENV_VAR, TEST_TOKEN)
    app.dependency_overrides[get_orchestrator] = lambda: test_orchestrator
    app.dependency_overrides[get_settings] = lambda: WorkerConfig()
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers():
    return {"Authorization": f"Bearer {TEST_TOKEN}"}


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


class TestTokenDisabled:
    def test_gated_path_is_open_when_token_unset(self, unauthenticated_client):
        assert unauthenticated_client.get(GATED_PATH).status_code == 200

    def test_openapi_is_open_when_token_unset(self, unauthenticated_client):
        assert unauthenticated_client.get("/openapi.json").status_code == 200

    def test_system_info_reports_auth_not_required(self, unauthenticated_client):
        response = unauthenticated_client.get("/system/info")

        assert response.status_code == 200
        assert response.json()["auth_required"] is False


class TestTokenEnabled:
    def test_missing_header_returns_401(self, client):
        response = client.get(GATED_PATH)

        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"
        assert response.json() == {"detail": "Missing or invalid API token"}

    def test_wrong_token_returns_401(self, client):
        response = client.get(GATED_PATH, headers={"Authorization": "Bearer wrong-token"})

        assert response.status_code == 401

    def test_wrong_scheme_returns_401(self, client):
        response = client.get(GATED_PATH, headers={"Authorization": f"Basic {TEST_TOKEN}"})

        assert response.status_code == 401

    def test_valid_token_is_accepted(self, client, auth_headers):
        assert client.get(GATED_PATH, headers=auth_headers).status_code == 200

    def test_openapi_is_gated(self, client, auth_headers):
        assert client.get("/openapi.json").status_code == 401
        assert client.get("/openapi.json", headers=auth_headers).status_code == 200

    @pytest.mark.parametrize("path", ["/health", "/health/ready", "/system/info"])
    def test_open_paths_need_no_token(self, client, path, monkeypatch):
        from chap_core.rest_api import common_routes

        monkeypatch.setattr(common_routes, "_check_db", lambda: "ok")
        monkeypatch.setattr(common_routes, "_check_redis", lambda: "ok")
        monkeypatch.setattr(common_routes, "_check_celery", lambda: "ok")

        assert client.get(path).status_code == 200

    def test_system_info_reports_auth_required(self, client):
        assert client.get("/system/info").json()["auth_required"] is True

    def test_401_keeps_cors_headers(self, client):
        response = client.get(GATED_PATH, headers={"Origin": "http://localhost:3000"})

        assert response.status_code == 401
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"

    def test_preflight_is_not_blocked(self, client):
        response = client.options(
            GATED_PATH,
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
        )

        assert response.status_code == 200


class TestServiceKeyHeaderCarriesTheToken:
    """servicekit can only send X-Service-Key, so the API token is accepted there too."""

    def test_api_token_in_service_key_header_is_accepted(self, client, sample_registration):
        response = client.post(
            "/v2/services/$register",
            json=sample_registration,
            headers={"X-Service-Key": TEST_TOKEN},
        )

        assert response.status_code == 200

    def test_wrong_value_in_service_key_header_is_rejected(self, client):
        response = client.get(GATED_PATH, headers={"X-Service-Key": "wrong-token"})

        assert response.status_code == 401

    def test_registration_key_is_accepted_on_the_service_registry(self, client, monkeypatch, sample_registration):
        monkeypatch.setenv(SERVICE_KEY_ENV_VAR, TEST_SERVICE_KEY)

        response = client.post(
            "/v2/services/$register",
            json=sample_registration,
            headers={"X-Service-Key": TEST_SERVICE_KEY},
        )

        assert response.status_code == 200

    def test_registration_key_is_not_a_general_api_credential(self, client, monkeypatch):
        monkeypatch.setenv(SERVICE_KEY_ENV_VAR, TEST_SERVICE_KEY)

        response = client.get("/openapi.json", headers={"X-Service-Key": TEST_SERVICE_KEY})

        assert response.status_code == 401


class TestWeakTokenWarning:
    def test_short_token_warns(self, monkeypatch, caplog):
        monkeypatch.setenv(API_TOKEN_ENV_VAR, "1")

        with caplog.at_level("WARNING"):
            warn_on_weak_token()

        assert API_TOKEN_ENV_VAR in caplog.text

    def test_long_token_does_not_warn(self, monkeypatch, caplog):
        monkeypatch.setenv(API_TOKEN_ENV_VAR, "a" * MIN_TOKEN_LENGTH)

        with caplog.at_level("WARNING"):
            warn_on_weak_token()

        assert caplog.text == ""

    def test_unset_token_does_not_warn(self, monkeypatch, caplog):
        monkeypatch.delenv(API_TOKEN_ENV_VAR, raising=False)

        with caplog.at_level("WARNING"):
            warn_on_weak_token()

        assert caplog.text == ""


class TestServiceRegistrationWithBothSecretsSet:
    @pytest.fixture(autouse=True)
    def service_key(self, monkeypatch):
        monkeypatch.setenv(SERVICE_KEY_ENV_VAR, TEST_SERVICE_KEY)

    def test_service_key_alone_is_accepted(self, client, sample_registration):
        """The chapkit path: servicekit can only send X-Service-Key."""
        response = client.post(
            "/v2/services/$register",
            json=sample_registration,
            headers={"X-Service-Key": TEST_SERVICE_KEY},
        )

        assert response.status_code == 200

    def test_api_token_alone_is_rejected(self, client, sample_registration, auth_headers):
        response = client.post("/v2/services/$register", json=sample_registration, headers=auth_headers)

        assert response.status_code == 401

    def test_both_secrets_are_accepted(self, client, sample_registration, auth_headers):
        response = client.post(
            "/v2/services/$register",
            json=sample_registration,
            headers={**auth_headers, "X-Service-Key": TEST_SERVICE_KEY},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "registered"
