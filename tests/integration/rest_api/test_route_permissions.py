import pytest
from fastapi import APIRouter, FastAPI

from chap_core.auth.permissions import Permission
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
