"""Opt-in shared-secret authentication.

Two independent secrets, each configured by environment variable and each disabled when
unset. ``CHAP_API_TOKEN`` gates the whole API, and ``SERVICEKIT_REGISTRATION_KEY`` gates
chapkit service registration. When both are configured, the registration endpoints
require both.

The API token is normally presented as ``Authorization: Bearer <token>``, but it is also
accepted in ``X-Service-Key`` because servicekit can only send that header. On the service
registry paths the registration key is accepted there as well, so a chapkit service holding
either secret can register without needing to send an ``Authorization`` header.

The token is enforced by the route dependencies below, which ``app.py`` attaches to every
router except the open health and info endpoints. FastAPI does not run route dependencies
for its own ``/docs``, ``/redoc`` and ``/openapi.json``, so the API reference stays public
and its Authorize button sends the token on "Try it out".
"""

import logging
import os
import secrets

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

API_TOKEN_ENV_VAR = "CHAP_API_TOKEN"
SERVICE_KEY_ENV_VAR = "SERVICEKIT_REGISTRATION_KEY"
SERVICE_KEY_HEADER = "X-Service-Key"

# Length of a `openssl rand -hex 32` token. The API has no rate limiting, so a short token
# is brute-forceable by anyone who can reach the port.
MIN_TOKEN_LENGTH = 32

# auto_error=False: a missing or non-Bearer header is answered by require_api_token, with
# the same 401 as a wrong token, and not at all when no token is configured.
bearer_scheme = HTTPBearer(
    auto_error=False,
    description=f"The server's {API_TOKEN_ENV_VAR}, when it has one. `/system/info` reports whether it does.",
)


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


def _authorize(
    request: Request, credentials: HTTPAuthorizationCredentials | None, accept_registration_key: bool
) -> None:
    expected = get_api_token()
    if expected is None:
        return
    if credentials is not None and secret_matches(credentials.credentials, expected):
        return
    service_key = request.headers.get(SERVICE_KEY_HEADER)
    if secret_matches(service_key, expected):
        return
    registration_key = get_service_key()
    if accept_registration_key and registration_key is not None and secret_matches(service_key, registration_key):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API token",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_api_token(
    request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)
) -> None:
    """Reject the request unless it carries the API token, when ``CHAP_API_TOKEN`` is set.

    ``X-Service-Key`` is read from the request rather than declared as a header parameter,
    so it does not show up as a parameter of every operation in the OpenAPI spec.
    """
    _authorize(request, credentials, accept_registration_key=False)


def require_api_token_or_registration_key(
    request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)
) -> None:
    """``require_api_token`` for the service registry, which also takes the registration key.

    servicekit can only send ``X-Service-Key``, never ``Authorization``, so self-registering
    chapkit services present their secret there instead. Confined to the service registry so a
    registration key cannot be used as a general-purpose API credential.
    """
    _authorize(request, credentials, accept_registration_key=True)
