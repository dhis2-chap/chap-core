"""Each route's permission, declared with FastAPI's ``Security`` and checked at startup.

``require(Permission.READ_DATASETS)`` on a route makes a caller without that permission get 403,
and puts the permission in the OpenAPI spec, where ``check_route_permissions`` reads it back.
"""

from typing import Any

from fastapi import Depends, FastAPI, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, SecurityScopes

from chap_core.auth.permissions import Permission
from chap_core.auth.principal import Principal
from chap_core.rest_api.auth import PUBLIC_PATHS, refusal, route_path

# Documents the bearer scheme in the spec and gives /docs its Authorize button. The
# authentication middleware does the checking, so a missing header is not an error here.
bearer = HTTPBearer(auto_error=False)

# Routes left out of the OpenAPI spec, where the startup check cannot see a declaration.
HIDDEN_ROUTE_PERMISSIONS: dict[str, Permission] = {
    "/v2/services/{service_id}/run/{path}": Permission.CALL_SERVICES,
}


class PermissionDenied(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def authorize(
    security_scopes: SecurityScopes,
    request: Request,
    _credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> Principal:
    """The caller, if it holds every permission the route declares; 403 otherwise."""
    principal = request.user
    if not isinstance(principal, Principal):
        raise RuntimeError("authorize needs the authentication middleware")
    missing = [scope for scope in security_scopes.scopes if scope not in principal.permissions]
    if missing:
        what = " and ".join(missing)
        permission = "permission" if len(missing) == 1 else "permissions"
        raise PermissionDenied(
            f"{request.method} {route_path(request)} needs the {what} {permission}, which {principal.id} does not have."
        )
    return principal


def require(permission: Permission) -> Any:
    """The dependency a route adds to need ``permission``."""
    return Security(authorize, scopes=[permission.value])


async def permission_denied_handler(_request: Request, exc: Exception):
    assert isinstance(exc, PermissionDenied)
    return refusal(403, "PermissionsInsufficient", exc.message)


class RoutePermissionError(RuntimeError):
    """A route declares no permission, or more than one."""


def check_route_permissions(app: FastAPI) -> None:
    """Refuse to start unless every operation is public or declares exactly one known permission."""
    problems = []
    for path, operations in app.openapi()["paths"].items():
        for method, operation in operations.items():
            scopes = {
                scope
                for requirement in operation.get("security", [])
                for scopes_of_scheme in requirement.values()
                for scope in scopes_of_scheme
            }
            name = f"{method.upper()} {path}"
            if path in PUBLIC_PATHS:
                if scopes:
                    problems.append(f"{name} is public but declares {', '.join(sorted(scopes))}")
            elif len(scopes) != 1:
                problems.append(f"{name} declares {len(scopes)} permissions, not one")
            elif not scopes <= set(Permission):
                problems.append(f"{name} declares the unknown permission {scopes.pop()}")
    if problems:
        raise RoutePermissionError("Routes without exactly one permission:\n" + "\n".join(problems))
