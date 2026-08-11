"""Opt-in shared-secret authentication.

Two independent secrets, each configured by environment variable and each disabled when
unset. ``CHAP_API_TOKEN`` gates the whole API, and ``SERVICEKIT_REGISTRATION_KEY`` gates
chapkit service registration. When both are configured, the registration endpoints
require both.

The API token is normally presented as ``Authorization: Bearer <token>``, but it is also
accepted in ``X-Service-Key`` because servicekit can only send that header. On the service
registry paths the registration key is accepted there as well, so a chapkit service holding
either secret can register without needing to send an ``Authorization`` header.
"""

import logging
import os
import secrets

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

API_TOKEN_ENV_VAR = "CHAP_API_TOKEN"
SERVICE_KEY_ENV_VAR = "SERVICEKIT_REGISTRATION_KEY"
SERVICE_KEY_HEADER = "X-Service-Key"

# Length of a `openssl rand -hex 32` token. The API has no rate limiting, so a short token
# is brute-forceable by anyone who can reach the port.
MIN_TOKEN_LENGTH = 32

# /health is called without headers by the container HEALTHCHECK and the Helm probes, and
# /system/info is how clients discover that a token is required.
OPEN_PATHS = frozenset({"/health", "/health/ready", "/system/info"})

# servicekit can only send X-Service-Key, never Authorization, so self-registering chapkit
# services present their secret there instead. Confined to the service registry so a
# registration key cannot be used as a general-purpose API credential.
SERVICE_REGISTRY_PREFIX = "/v2/services"


def get_api_token() -> str | None:
    """The configured API token, or None when authentication is disabled."""
    return os.getenv(API_TOKEN_ENV_VAR) or None


def warn_on_weak_token() -> None:
    """Log a warning at startup if the configured API token is too short to be safe."""
    token = get_api_token()
    if token is not None and len(token) < MIN_TOKEN_LENGTH:
        logger.warning(
            "%s is only %d characters and is easily guessed. Use at least %d characters, "
            "e.g. `openssl rand -hex 32` or `uuidgen`.",
            API_TOKEN_ENV_VAR,
            len(token),
            MIN_TOKEN_LENGTH,
        )


def get_service_key() -> str | None:
    """The configured service registration key, or None when it is disabled."""
    return os.getenv(SERVICE_KEY_ENV_VAR) or None


def secret_matches(presented: str | None, expected: str) -> bool:
    """Timing-safe comparison of a presented secret against the configured one."""
    if not presented:
        return False
    # compare_digest on str raises TypeError for non-ascii, so compare encoded bytes.
    return secrets.compare_digest(presented.encode(), expected.encode())


def _bearer_matches(header: str | None, expected: str) -> bool:
    if not header:
        return False
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer":
        return False
    return secret_matches(token, expected)


class ApiTokenMiddleware:
    """Reject requests without a valid bearer token when ``CHAP_API_TOKEN`` is set.

    Raw ASGI rather than ``BaseHTTPMiddleware`` so the streaming proxy in
    ``v2/routers/proxy.py`` passes through untouched, and so ``/docs`` and ``/openapi.json``
    are covered too -- route dependencies never reach those.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        expected = get_api_token()
        path = self._path(scope)
        if expected is None or path in OPEN_PATHS:
            return await self.app(scope, receive, send)

        if not self._is_authorized(scope, path, expected):
            response = JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid API token"},
                headers={"WWW-Authenticate": "Bearer"},
            )
            return await response(scope, receive, send)

        return await self.app(scope, receive, send)

    @classmethod
    def _is_authorized(cls, scope: Scope, path: str, api_token: str) -> bool:
        if _bearer_matches(cls._header(scope, b"authorization"), api_token):
            return True

        service_key = cls._header(scope, b"x-service-key")
        if not service_key:
            return False
        if secret_matches(service_key, api_token):
            return True

        registration_key = get_service_key()
        if registration_key is not None and path.startswith(SERVICE_REGISTRY_PREFIX):
            return secret_matches(service_key, registration_key)
        return False

    @staticmethod
    def _path(scope: Scope) -> str:
        path: str = scope.get("path", "")
        root_path: str = scope.get("root_path", "")
        if root_path and path.startswith(root_path):
            return path[len(root_path) :] or "/"
        return path

    @staticmethod
    def _header(scope: Scope, name: bytes) -> str | None:
        for key, value in scope.get("headers", []):
            if key == name:
                return str(value.decode("latin-1"))
        return None
