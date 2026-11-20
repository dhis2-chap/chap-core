"""Every route enforces the permission it declares, not only declares it.

Each permission gets a token whose tier holds that permission alone. Every route the app serves,
including those hidden from the OpenAPI spec, is called with a token lacking its permission,
which must get 403, and with the token holding only that permission, which must get past the
check.
"""

import re

import pytest
from fastapi.testclient import TestClient

from chap_core.auth.permissions import Permission
from chap_core.rest_api import common_routes
from chap_core.rest_api.app import app
from chap_core.rest_api.auth import PUBLIC_PATHS
from chap_core.rest_api.authorization import declared_permissions, served_routes
from chap_core.rest_api.v1.routers.dependencies import get_settings
from chap_core.rest_api.v2.dependencies import get_orchestrator
from chap_core.rest_api.worker_functions import WorkerConfig


def tier_name(permission: Permission) -> str:
    return permission.value.replace(":", "-")


def secret(permission: Permission) -> str:
    return f"{tier_name(permission)}-".ljust(40, "x")


def gated_operations() -> list[tuple[str, str, str]]:
    operations = []
    for path, methods, dependant in served_routes(app):
        if dependant is None or path in PUBLIC_PATHS:
            continue
        (permission,) = declared_permissions(dependant)
        for method in sorted(methods - {"HEAD"}):
            operations.append((method, path, permission))
    return operations


OPERATIONS = gated_operations()


@pytest.fixture
def client(test_orchestrator, monkeypatch, tmp_path):
    tiers = "\n".join(f"    {tier_name(p)}:\n      permissions: [{p.value}]" for p in Permission)
    tokens = "\n".join(
        f"    - {{id: {tier_name(p)}, token_env: CHAP_TEST_{tier_name(p).upper().replace('-', '_')}, tier: {tier_name(p)}}}"
        for p in Permission
    )
    config = tmp_path / "chap-server.yaml"
    config.write_text(f"authentication:\n  tiers:\n{tiers}\n  tokens:\n{tokens}\n")
    monkeypatch.setenv("CHAP_CONFIG", str(config))
    monkeypatch.delenv("CHAP_API_TOKEN", raising=False)
    for permission in Permission:
        monkeypatch.setenv(f"CHAP_TEST_{tier_name(permission).upper().replace('-', '_')}", secret(permission))
    # The probe would wait for a broker; the permission check is what is under test.
    monkeypatch.setattr(common_routes, "_check_celery_db_roundtrip", lambda: "ok")
    app.dependency_overrides[get_orchestrator] = lambda: test_orchestrator
    app.dependency_overrides[get_settings] = lambda: WorkerConfig()
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


def call(client: TestClient, method: str, path: str, permission: Permission):
    url = re.sub(r"\{[^}]+\}", "1", path)
    return client.request(method, url, headers={"Authorization": f"Bearer {secret(permission)}"})


def test_the_sweep_covers_every_permission():
    assert {permission for _, _, permission in OPERATIONS} == {permission.value for permission in Permission}


@pytest.mark.parametrize(("method", "path", "permission"), OPERATIONS)
def test_a_token_without_the_permission_is_refused(client, method, path, permission):
    other = next(p for p in Permission if p.value != permission)

    response = call(client, method, path, other)

    assert response.status_code == 403, response.text
    assert permission in response.json()["detail"]


@pytest.mark.parametrize(("method", "path", "permission"), OPERATIONS)
def test_a_token_with_only_the_permission_gets_past_the_check(client, method, path, permission):
    response = call(client, method, path, Permission(permission))

    assert response.status_code not in (401, 403), response.text
