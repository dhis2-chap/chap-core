"""Conformance tests for docs/contributor/auth_contract.md.

The first classes record what chap-core did before named tokens, so the auth rework keeps it.
``test_api_token.py`` covers ``CHAP_API_TOKEN`` on its own and with the registration key.
"""

import pytest
from fastapi.testclient import TestClient

from chap_core.auth.permissions import READ, Permission
from chap_core.rest_api.app import app
from tests.integration.rest_api.db_endpoint_helpers import create_make_data_request
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
    config = tmp_path / "chap-server.yml"
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


class TestPermissions:
    def read_headers(self):
        return {"Authorization": f"Bearer {READ_TOKEN}"}

    def test_a_read_token_reads(self, named_client):
        assert named_client.get("/v1/terminology", headers=self.read_headers()).status_code == 200

    def test_a_token_without_the_permission_gets_403_naming_it_and_the_token(self, named_client):
        response = named_client.put("/v1/settings", json={}, headers=self.read_headers())

        assert response.status_code == 403
        assert response.json()["code"] == "PermissionsInsufficient"
        assert response.json()["detail"] == (
            "PUT /v1/settings needs the write:settings permission, which token:dashboard does not have."
        )

    def test_the_read_tier_cannot_read_jobs(self, named_client):
        assert named_client.get("/v1/jobs", headers=self.read_headers()).status_code == 403

    def test_the_hidden_proxy_route_needs_call_services(self, named_client):
        assert named_client.get("/v2/services/any/run/info", headers=self.read_headers()).status_code == 403

    def test_the_operator_tier_passes_the_permission_check(self, named_client):
        response = named_client.put("/v1/settings", json={}, headers={"Authorization": f"Bearer {OPERATOR_TOKEN}"})

        assert response.status_code != 403


class TestMe:
    def test_with_authentication_off_the_caller_is_open_with_every_permission(self, make_client):
        me = make_client().get("/v1/auth/me").json()

        assert (me["id"], me["tier"]) == ("open", "operator")
        assert set(me["permissions"]) == {permission.value for permission in Permission}

    def test_a_named_token_reports_its_id_tier_and_permissions(self, named_client):
        me = named_client.get("/v1/auth/me", headers={"Authorization": f"Bearer {READ_TOKEN}"}).json()

        assert (me["id"], me["tier"]) == ("token:dashboard", "read")
        assert me["permissions"] == sorted(permission.value for permission in READ)

    def test_without_a_token_the_caller_is_anonymous(self, named_client):
        response = named_client.get("/v1/auth/me")

        assert response.status_code == 200
        assert response.json() == {"id": "anonymous", "tier": "public", "permissions": []}

    def test_the_api_token_is_the_legacy_operator(self, make_client):
        client = make_client(**{API_TOKEN_ENV_VAR: API_TOKEN})

        me = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {API_TOKEN}"}).json()

        assert (me["id"], me["tier"]) == ("token:legacy", "operator")


class TestRefusalsShareOneBody:
    def test_the_registration_key_refusal_has_the_contract_body(self, make_client, sample_registration):
        client = make_client(**{API_TOKEN_ENV_VAR: API_TOKEN, SERVICE_KEY_ENV_VAR: REGISTRATION_KEY})

        response = register(client, sample_registration, {"Authorization": f"Bearer {API_TOKEN}"})

        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"
        assert response.json()["code"] == "TokenInvalid"
        assert response.json()["detail"] == "Missing or invalid service key"

    def test_me_answers_anonymous_to_an_invalid_token(self, named_client):
        response = named_client.get("/v1/auth/me", headers={"Authorization": "Bearer " + "x" * 32})

        assert response.json()["id"] == "anonymous"


RUNNER_TOKEN = "u" * 32
SETUPS_TOKEN = "p" * 32
JOBS_TOKEN = "j" * 32

NARROW_TIERS = """
authentication:
  tiers:
    runner:
      permissions: [run:backtests, run:predictions]
    setups:
      permissions: [write:prediction-setups]
    jobs:
      permissions: [read:jobs]
  tokens:
    - {id: runner, token_env: CHAP_TEST_RUNNER_TOKEN, tier: runner}
    - {id: setups, token_env: CHAP_TEST_SETUPS_TOKEN, tier: setups}
    - {id: jobs, token_env: CHAP_TEST_JOBS_TOKEN, tier: jobs}
    - {id: dashboard, token_env: CHAP_TEST_READ_TOKEN, tier: read}
"""


@pytest.fixture
def narrow_client(make_client, tmp_path):
    config = tmp_path / "chap-server.yml"
    config.write_text(NARROW_TIERS)
    return make_client(
        CHAP_CONFIG=str(config),
        CHAP_TEST_RUNNER_TOKEN=RUNNER_TOKEN,
        CHAP_TEST_SETUPS_TOKEN=SETUPS_TOKEN,
        CHAP_TEST_JOBS_TOKEN=JOBS_TOKEN,
        CHAP_TEST_READ_TOKEN=READ_TOKEN,
    )


class TestSecondPermissions:
    @pytest.mark.parametrize(
        ("endpoint", "extra"),
        [
            ("create-backtest-with-data/", {"n_periods": 3, "n_splits": 2, "stride": 1}),
            ("make-prediction", {"n_periods": 3}),
        ],
    )
    def test_inline_data_routes_also_need_write_datasets(self, narrow_client, dataset_make_request, endpoint, extra):
        payload = dataset_make_request.model_dump(mode="json") | {"model_id": "naive_model", **extra}

        response = narrow_client.post(
            f"/v1/analytics/{endpoint}", json=payload, headers={"Authorization": f"Bearer {RUNNER_TOKEN}"}
        )

        assert response.status_code == 403
        assert "write:datasets" in response.json()["detail"]

    def test_changing_a_setups_alert_policy_also_needs_write_alerts(self, narrow_client):
        response = narrow_client.patch(
            "/v1/crud/prediction-setups/1",
            json={"alertPolicyId": 1},
            headers={"Authorization": f"Bearer {SETUPS_TOKEN}"},
        )

        assert response.status_code == 403
        assert "write:alerts" in response.json()["detail"]

    def test_the_read_tier_cannot_download_a_datasets_observations(self, narrow_client):
        response = narrow_client.get("/v1/crud/datasets/1/csv", headers={"Authorization": f"Bearer {READ_TOKEN}"})

        assert response.status_code == 403
        assert "export:datasets" in response.json()["detail"]

    def test_a_jobs_stored_request_also_needs_export_datasets(self, narrow_client):
        """The stored request can hold the observations a job posted."""
        response = narrow_client.get("/v1/jobs/job-1/request", headers={"Authorization": f"Bearer {JOBS_TOKEN}"})

        assert response.status_code == 403
        assert "export:datasets" in response.json()["detail"]


def test_running_a_prediction_setup_also_needs_write_datasets(narrow_client, example_polygons):
    request = create_make_data_request(example_polygons, ["rainfall", "disease_cases", "population"])
    payload = request.model_dump(mode="json", exclude={"data_sources", "masked_ranges", "target_column"})
    payload.pop("provided_data_columns", None)

    response = narrow_client.post(
        "/v1/crud/prediction-setups/1/run", json=payload, headers={"Authorization": f"Bearer {RUNNER_TOKEN}"}
    )

    assert response.status_code == 403, response.text
    assert "write:datasets" in response.json()["detail"]


def test_one_secret_for_both_keys_still_registers_a_service(make_client, sample_registration):
    """Older docs told chapkit installs to set SERVICEKIT_REGISTRATION_KEY to CHAP_API_TOKEN."""
    client = make_client(**{API_TOKEN_ENV_VAR: API_TOKEN, SERVICE_KEY_ENV_VAR: API_TOKEN})

    assert register(client, sample_registration, {"X-Service-Key": API_TOKEN}).status_code == 200
    assert client.get("/v1/terminology", headers={"Authorization": f"Bearer {API_TOKEN}"}).status_code == 200
