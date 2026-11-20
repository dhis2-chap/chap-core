"""docs/contributor/auth_contract.md states what the code does."""

import re
from pathlib import Path

import yaml

from chap_core.auth.permissions import AUTHENTICATED, OPERATOR, READ, Permission
from chap_core.auth.tokens import resolve_tiers
from chap_core.settings import ChapSettings

CONTRACT = (Path(__file__).parents[2] / "docs" / "contributor" / "auth_contract.md").read_text()


def permission_table() -> dict[str, tuple[bool, bool, bool]]:
    rows = re.findall(
        r"^\| `([a-z]+:[a-z-]+)` \| [^|]+\|\s*(yes)?\s*\|\s*(yes)?\s*\|\s*(yes)?\s*\|$", CONTRACT, re.MULTILINE
    )
    return {
        permission: (read == "yes", authenticated == "yes", operator == "yes")
        for permission, read, authenticated, operator in rows
    }


def test_the_permission_table_lists_every_permission_and_its_tiers():
    expected = {p.value: (p in READ, p in AUTHENTICATED, p in OPERATOR) for p in Permission}

    assert permission_table() == expected


def test_the_tier_example_is_valid():
    block = re.search(r"```yaml\n(authentication:\n  tiers:\n    caps:.*?)```", CONTRACT, re.DOTALL)
    assert block is not None

    settings = ChapSettings(**yaml.safe_load(block.group(1)))

    assert resolve_tiers(settings.authentication.tiers)["caps"] == READ | {
        Permission.WRITE_DATASETS,
        Permission.RUN_PREDICTIONS,
        Permission.READ_JOBS,
    }


def test_the_route_table_matches_the_app():
    from chap_core.rest_api.app import app
    from chap_core.rest_api.auth import PUBLIC_PATHS
    from chap_core.rest_api.authorization import declared_permissions, served_routes

    documented = {
        (method, path): permission.strip("`")
        for method, path, permission in re.findall(
            r"^\| ([A-Z]+) \| `([^`]+)` \| (public|`[a-z]+:[a-z-]+`) \|$", CONTRACT, re.MULTILINE
        )
    }
    served = {}
    for path, methods, dependant in served_routes(app):
        public = path in PUBLIC_PATHS
        permission = "public" if public or dependant is None else declared_permissions(dependant).pop()
        for method in methods - {"HEAD"}:
            served[(method, path)] = permission
    # The API reference pages are Starlette routes, documented as one group under the public tier.
    served = {
        key: value
        for key, value in served.items()
        if key[1] not in {"/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"}
    }

    assert documented == served
