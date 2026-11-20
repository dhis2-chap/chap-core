import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from chap_core.auth.permissions import Permission
from chap_core.auth.tokens import AuthenticationConfigError
from chap_core.rest_api.app import app
from chap_core.rest_api.authorization import RoutePermissionError, check_route_permissions, require


def test_every_route_of_the_app_declares_one_permission():
    check_route_permissions(app)


def test_a_route_without_a_permission_stops_startup():
    bare = FastAPI()

    @bare.get("/v1/unguarded")
    def unguarded():
        return {}

    with pytest.raises(RoutePermissionError, match="GET /v1/unguarded declares 0 permissions"):
        check_route_permissions(bare)


def test_a_route_with_two_permissions_stops_startup():
    router = APIRouter(dependencies=[require(Permission.READ_DATASETS)])

    @router.get("/v1/twice", dependencies=[require(Permission.READ_MODELS)])
    def twice():
        return {}

    bare = FastAPI()
    bare.include_router(router)

    with pytest.raises(RoutePermissionError, match="GET /v1/twice declares 2 permissions"):
        check_route_permissions(bare)


def test_a_public_route_may_not_declare_a_permission():
    bare = FastAPI()

    @bare.get("/health", dependencies=[require(Permission.READ_SYSTEM)])
    def health():
        return {}

    with pytest.raises(RoutePermissionError, match="GET /health is public"):
        check_route_permissions(bare)


def test_a_route_hidden_from_the_spec_is_checked_too():
    bare = FastAPI()

    @bare.get("/v1/hidden", include_in_schema=False)
    def hidden():
        return {}

    with pytest.raises(RoutePermissionError, match="GET /v1/hidden declares 0 permissions"):
        check_route_permissions(bare)


def test_startup_refuses_a_token_whose_secret_is_unset(monkeypatch, tmp_path):
    config = tmp_path / "chap-server.yaml"
    config.write_text("authentication:\n  tokens:\n    - {id: caps, token_env: CHAP_TEST_UNSET, tier: operator}\n")
    monkeypatch.setenv("CHAP_CONFIG", str(config))
    monkeypatch.delenv("CHAP_TEST_UNSET", raising=False)

    with pytest.raises(AuthenticationConfigError, match="CHAP_TEST_UNSET is not set"), TestClient(app):
        pass
