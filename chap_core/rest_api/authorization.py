"""Each route's permission, declared with FastAPI's ``Security`` and checked at startup.

``require(Permission.READ_DATASETS)`` on a route makes a caller without that permission get 403,
and puts the permission in the OpenAPI spec, where ``check_route_permissions`` reads it back.
"""

from collections.abc import Iterator
from typing import Any

from fastapi import Depends, FastAPI, Request, Security
from fastapi.dependencies.models import Dependant
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, SecurityScopes

from chap_core.auth.permissions import Permission
from chap_core.auth.principal import Principal
from chap_core.rest_api.auth import PUBLIC_PATHS, refusal, route_path

# Documents the bearer scheme in the spec and gives /docs its Authorize button. The
# authentication middleware does the checking, so a missing header is not an error here.
bearer = HTTPBearer(auto_error=False)


class Refusal(Exception):
    """A request refused after authentication, answered with the contract's refusal body."""

    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class PermissionDenied(Refusal):
    def __init__(self, message: str):
        super().__init__(403, "PermissionsInsufficient", message)


async def authorize(
    security_scopes: SecurityScopes,
    request: Request,
    _credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> Principal:
    """The caller, if it holds every permission the route declares; 403 otherwise.

    Async because it does no blocking work: a sync dependency would take a thread from the pool
    that ``app.py`` caps at the number of database connections, even on async routes.
    """
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


def caller(request: Request) -> Principal:
    """The principal the authentication middleware resolved for this request."""
    principal = request.user
    if not isinstance(principal, Principal):
        raise RuntimeError("caller needs the authentication middleware")
    return principal


def ensure(request: Request, permission: Permission, why: str) -> None:
    """Refuse with 403 unless the caller also holds ``permission``, for a route that does two things."""
    principal = caller(request)
    if permission not in principal.permissions:
        raise PermissionDenied(
            f"{request.method} {route_path(request)} {why}, which needs the {permission.value} permission, "
            f"which {principal.id} does not have."
        )


def require(permission: Permission) -> Any:
    """The dependency a route adds to need ``permission``."""
    return Security(authorize, scopes=[permission.value])


async def refusal_handler(_request: Request, exc: Exception):
    assert isinstance(exc, Refusal)
    return refusal(exc.status_code, exc.code, exc.message)


class RoutePermissionError(RuntimeError):
    """A route declares no permission, or more than one."""


def declared_permissions(dependant: Dependant) -> set[str]:
    """The permissions a route's dependencies declare through ``require``."""
    scopes = set(dependant.own_oauth_scopes or []) if dependant.call is authorize else set()
    for sub_dependant in dependant.dependencies:
        scopes |= declared_permissions(sub_dependant)
    return scopes


def served_routes(app: FastAPI) -> Iterator[tuple[str, set[str], Dependant | None]]:
    """Every path the app serves, with its methods and dependencies, including routes hidden from the spec."""
    for route in app.routes:
        # Included routers are resolved lazily; each context is one route with its full path.
        contexts = route.effective_route_contexts() if hasattr(route, "effective_route_contexts") else [route]
        for context in contexts:
            path = getattr(context, "path", "")
            yield path, set(getattr(context, "methods", None) or []), getattr(context, "dependant", None)


def check_route_permissions(app: FastAPI) -> None:
    """Refuse to start unless every route is public or declares exactly one known permission."""
    problems = []
    for path, methods, dependant in served_routes(app):
        name = f"{','.join(sorted(methods))} {path}"
        scopes = declared_permissions(dependant) if dependant is not None else set()
        if path in PUBLIC_PATHS:
            if scopes:
                problems.append(f"{name} is public but declares {', '.join(sorted(scopes))}")
        elif len(scopes) != 1:
            problems.append(f"{name} declares {len(scopes)} permissions, not one")
        elif not scopes <= set(Permission):
            problems.append(f"{name} declares the unknown permission {scopes.pop()}")
    if problems:
        raise RoutePermissionError("Routes without exactly one permission:\n" + "\n".join(problems))
