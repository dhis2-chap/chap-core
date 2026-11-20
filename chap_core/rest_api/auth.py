"""Authentication for the REST API: who is calling, decided before the request body is read.

The contract is docs/contributor/auth_contract.md. Authentication is on when ``CHAP_API_TOKEN``
is set or ``chap-server.yaml`` lists named tokens; ``CHAP_API_TOKEN`` is then an operator token
with the id ``legacy``. ``SERVICEKIT_REGISTRATION_KEY`` gates the service registration routes
on its own, through ``verify_service_key``.
"""

import logging
import time

from starlette.authentication import AuthCredentials, AuthenticationBackend, AuthenticationError
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse, Response

from chap_core.auth.principal import ANONYMOUS, OPEN, Principal
from chap_core.auth.tokens import MIN_SECRET_LENGTH, authenticate, get_authentication
from chap_core.settings import get_settings

logger = logging.getLogger(__name__)

API_TOKEN_ENV_VAR = "CHAP_API_TOKEN"  # noqa: S105 — name of the env var, not a secret
SERVICE_KEY_ENV_VAR = "SERVICEKIT_REGISTRATION_KEY"
SERVICE_KEY_HEADER = "X-Service-Key"

MIN_TOKEN_LENGTH = MIN_SECRET_LENGTH

# /health is called without headers by the container HEALTHCHECK and orchestrator probes, and
# /system/info is how clients discover that a token is required.
OPEN_PATHS = frozenset({"/health", "/health/ready", "/system/info"})

# FastAPI's own API reference, at its default URLs. It describes an open-source API, so it is
# public; the spec declares the bearer scheme, which gives /docs its Authorize button.
API_REFERENCE_PATHS = frozenset({"/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"})

# /v1/auth/me answers any caller, so a client can learn what its token may do.
PUBLIC_PATHS = OPEN_PATHS | API_REFERENCE_PATHS | {"/favicon.ico", "/v1/auth/me"}


def get_api_token() -> str | None:
    """The configured API token, or None when it is not set."""
    token = get_settings().api_token
    return token.get_secret_value() if token is not None else None


def warn_on_weak_token() -> None:
    """Log a warning at startup if ``CHAP_API_TOKEN`` is too short to be safe.

    Named tokens refuse a short secret; the legacy token keeps warning, so no install breaks.
    """
    token = get_api_token()
    if token is not None and len(token) < MIN_TOKEN_LENGTH:
        logger.warning(
            "%s is shorter than %d characters and is easily guessed. Use `openssl rand -hex 32`.",
            API_TOKEN_ENV_VAR,
            MIN_TOKEN_LENGTH,
        )


def get_service_key() -> str | None:
    """The configured service registration key, or None when it is disabled."""
    key = get_settings().service_registration_key
    return key.get_secret_value() if key is not None else None


class AuthenticationRefused(AuthenticationError):
    """A request refused with 401, with the openEO error code Open Geo Service also answers."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def refusal(status_code: int, code: str, message: str) -> JSONResponse:
    """The body every auth refusal has: FastAPI's ``detail`` plus the openEO error fields."""
    headers = {"WWW-Authenticate": "Bearer"} if status_code == 401 else None
    content = {"id": "authentication", "code": code, "message": message, "detail": message}
    return JSONResponse(status_code=status_code, content=content, headers=headers)


def on_authentication_error(_conn: HTTPConnection, exc: AuthenticationError) -> Response:
    if isinstance(exc, AuthenticationRefused):
        return refusal(401, exc.code, exc.message)
    return refusal(401, "TokenInvalid", str(exc))


_FAILURE_LOG_INTERVAL = 60.0
_failures_since_log = 0
_last_failure_log = float("-inf")


def _log_failure() -> None:
    """Log rejected credentials at most once a minute, so a client retrying a bad token cannot flood the log."""
    global _failures_since_log, _last_failure_log
    _failures_since_log += 1
    now = time.monotonic()
    if now - _last_failure_log >= _FAILURE_LOG_INTERVAL:
        logger.warning("Rejected %d request(s) with an invalid credential", _failures_since_log)
        _failures_since_log = 0
        _last_failure_log = now


def route_path(conn: HTTPConnection) -> str:
    """The path as the routes see it, so a deployment prefix cannot hide a gated route."""
    path: str = conn.scope.get("path", "")
    root_path: str = conn.scope.get("root_path", "")
    if root_path and path.startswith(root_path):
        return path[len(root_path) :] or "/"
    return path


def credentials(principal: Principal) -> tuple[AuthCredentials, Principal]:
    return AuthCredentials(sorted(principal.permissions)), principal


class ChapTokenBackend(AuthenticationBackend):
    """Resolve the caller from ``Authorization`` or ``X-Service-Key``.

    Answers 401 for an invalid credential, and for a missing one on any route but the public
    ones. Whether the caller has the route's permission is decided by the route (``authorize``).
    """

    async def authenticate(self, conn: HTTPConnection) -> tuple[AuthCredentials, Principal]:
        authentication = get_authentication()
        if not authentication.enabled:
            return credentials(OPEN)
        path = route_path(conn)
        principal = authenticate(
            conn.headers.get("authorization"), conn.headers.get(SERVICE_KEY_HEADER), path, authentication
        )
        if path in PUBLIC_PATHS:
            # Never refused, so a health check behind a proxy that adds a stale header still works.
            return credentials(principal or ANONYMOUS)
        if principal is None:
            _log_failure()
            raise AuthenticationRefused("TokenInvalid", "The credential is not valid for this Chap.")
        if principal is ANONYMOUS and conn.scope.get("method") != "OPTIONS":
            method = conn.scope.get("method", "")
            raise AuthenticationRefused("AuthenticationRequired", f"{method} {path} needs a token.")
        return credentials(principal)
