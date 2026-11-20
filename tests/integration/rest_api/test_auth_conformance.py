"""Conformance tests for docs/contributor/auth_contract.md.

The first classes record what chap-core did before named tokens, so the auth rework keeps it.
``test_api_token.py`` covers ``CHAP_API_TOKEN`` on its own and with the registration key.
"""

import pytest
from fastapi.testclient import TestClient

from chap_core.rest_api.app import app
from chap_core.rest_api.auth import API_TOKEN_ENV_VAR, SERVICE_KEY_ENV_VAR
from chap_core.rest_api.v1.routers.dependencies import get_settings
from chap_core.rest_api.v2.dependencies import get_orchestrator
from chap_core.rest_api.worker_functions import WorkerConfig

API_TOKEN = "a" * 32
REGISTRATION_KEY = "r" * 32

# Routes that answer without a database or a broker.
UNGATED_READS = ["/v2/services", "/v1/terminology", "/v1/analytics/weather-providers"]


@pytest.fixture
def make_client(test_orchestrator, monkeypatch):
    """A test client for the app with exactly the given secrets configured."""

    def make(**environment: str) -> TestClient:
        for name in (API_TOKEN_ENV_VAR, SERVICE_KEY_ENV_VAR, "CHAP_CONFIG"):
            monkeypatch.delenv(name, raising=False)
        for name, value in environment.items():
            monkeypatch.setenv(name, value)
        app.dependency_overrides[get_orchestrator] = lambda: test_orchestrator
        app.dependency_overrides[get_settings] = lambda: WorkerConfig()
        return TestClient(app, raise_server_exceptions=False)

    yield make
    app.dependency_overrides.clear()


def register(client: TestClient, registration: dict, headers: dict | None = None):
    return client.post("/v2/services/$register", json=registration, headers=headers or {})


class TestNothingConfigured:
    @pytest.mark.parametrize("path", UNGATED_READS)
    def test_every_route_is_open(self, make_client, path):
        assert make_client().get(path).status_code == 200

    def test_a_wrong_token_is_ignored(self, make_client):
        response = make_client().get("/v2/services", headers={"Authorization": "Bearer wrong"})

        assert response.status_code == 200

    def test_registration_needs_no_key(self, make_client, sample_registration):
        assert register(make_client(), sample_registration).status_code == 200


class TestRegistrationKeyOnly:
    @pytest.fixture
    def client(self, make_client):
        return make_client(**{SERVICE_KEY_ENV_VAR: REGISTRATION_KEY})

    def test_registration_needs_the_key(self, client, sample_registration):
        assert register(client, sample_registration).status_code == 401

    def test_registration_accepts_the_key(self, client, sample_registration):
        response = register(client, sample_registration, {"X-Service-Key": REGISTRATION_KEY})

        assert response.status_code == 200

    def test_ping_and_deregister_need_the_key(self, client, sample_registration):
        service_id = register(client, sample_registration, {"X-Service-Key": REGISTRATION_KEY}).json()["id"]

        assert client.put(f"/v2/services/{service_id}/$ping").status_code == 401
        assert client.delete(f"/v2/services/{service_id}").status_code == 401
        assert (
            client.delete(f"/v2/services/{service_id}", headers={"X-Service-Key": REGISTRATION_KEY}).status_code == 204
        )

    @pytest.mark.parametrize("path", UNGATED_READS)
    def test_the_rest_is_open(self, client, path):
        assert client.get(path).status_code == 200


class TestApiToken:
    @pytest.fixture
    def client(self, make_client):
        return make_client(**{API_TOKEN_ENV_VAR: API_TOKEN})

    @pytest.mark.parametrize(
        "headers",
        [
            {"Authorization": f"Bearer {API_TOKEN}"},
            {"Authorization": f"ApiToken {API_TOKEN}"},
            {"Authorization": f"bearer {API_TOKEN}"},
            {"X-Service-Key": API_TOKEN},
        ],
    )
    def test_the_token_is_accepted_three_ways(self, client, headers):
        assert client.get("/v1/terminology", headers=headers).status_code == 200

    def test_a_missing_token_gets_401_with_the_bearer_challenge(self, client):
        response = client.get("/v1/terminology")

        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"
        assert "detail" in response.json()

    def test_a_wrong_token_gets_401(self, client):
        assert client.get("/v1/terminology", headers={"Authorization": "Bearer " + "b" * 32}).status_code == 401

    def test_the_registration_key_also_opens_the_service_registry(self, make_client):
        client = make_client(**{API_TOKEN_ENV_VAR: API_TOKEN, SERVICE_KEY_ENV_VAR: REGISTRATION_KEY})

        assert client.get("/v2/services", headers={"X-Service-Key": REGISTRATION_KEY}).status_code == 200
        assert client.get("/v1/terminology", headers={"X-Service-Key": REGISTRATION_KEY}).status_code == 401


READ_TOKEN = "d" * 32
OPERATOR_TOKEN = "o" * 32

# An Open Geo Service block, plus a Chap-only tier.
SERVER_CONFIG = """
authentication:
  tokens:
    - id: dashboard
      token_env: CHAP_TEST_READ_TOKEN
      tier: read
    - id: caps
      token_env: CHAP_TEST_OPERATOR_TOKEN
      tier: operator
"""


@pytest.fixture
def named_client(make_client, tmp_path):
    config = tmp_path / "chap-server.yaml"
    config.write_text(SERVER_CONFIG)
    return make_client(
        CHAP_CONFIG=str(config), CHAP_TEST_READ_TOKEN=READ_TOKEN, CHAP_TEST_OPERATOR_TOKEN=OPERATOR_TOKEN
    )


class TestNamedTokens:
    def test_no_token_gets_401_with_the_bearer_challenge(self, named_client):
        response = named_client.get("/v1/terminology")

        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"
        assert response.json()["code"] == "AuthenticationRequired"

    def test_a_wrong_token_gets_401(self, named_client):
        response = named_client.get("/v1/terminology", headers={"Authorization": "Bearer " + "x" * 32})

        assert response.status_code == 401
        assert response.json()["code"] == "TokenInvalid"

    @pytest.mark.parametrize("scheme", ["Bearer", "ApiToken"])
    def test_a_named_token_is_accepted(self, named_client, scheme):
        response = named_client.get("/v1/terminology", headers={"Authorization": f"{scheme} {OPERATOR_TOKEN}"})

        assert response.status_code == 200

    def test_a_wrong_token_is_ignored_on_a_public_route(self, named_client):
        assert named_client.get("/health", headers={"Authorization": "Bearer wrong"}).status_code == 200

    def test_the_body_is_not_read_without_a_valid_token(self, named_client):
        """Malformed JSON gets 422 once the body is parsed, so 401 shows it never was."""
        headers = {"Authorization": "Bearer wrong", "Content-Type": "application/json"}

        assert named_client.post("/v1/crud/datasets", content=b"{not json", headers=headers).status_code == 401

    def test_system_info_reports_that_a_token_is_required(self, named_client):
        assert named_client.get("/system/info").json()["auth_required"] is True
