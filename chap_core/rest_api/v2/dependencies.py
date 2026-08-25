from functools import lru_cache

import httpx
from fastapi import Header, HTTPException, status

from chap_core.rest_api.auth import SERVICE_KEY_HEADER, get_service_key, secret_matches
from chap_core.rest_api.services.orchestrator import Orchestrator
from chap_core.util import load_redis

# Generous read/write timeouts: proxied artifact downloads can stream large payloads.
# Train/predict return 202 immediately, so they don't need a long window here.
_PROXY_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=10.0)


@lru_cache
def get_redis():
    return load_redis(db=3)


def get_orchestrator():
    return Orchestrator(redis_client=get_redis())


@lru_cache
def get_http_client() -> httpx.AsyncClient:
    """Shared AsyncClient used to proxy requests to registered chapkit services.

    Provided as a dependency so tests can override it with an ASGITransport-backed
    client pointed at a stub service.
    """
    return httpx.AsyncClient(timeout=_PROXY_TIMEOUT)


def verify_service_key(
    x_service_key: str | None = Header(default=None, alias=SERVICE_KEY_HEADER),
) -> str | None:
    """
    Verify the service registration API key.

    If SERVICEKIT_REGISTRATION_KEY is not configured, authentication is skipped.

    Raises:
        HTTPException 401: If the key is configured and the header is missing or wrong
    """
    expected_key = get_service_key()

    # If no key configured on server, skip authentication
    if expected_key is None:
        return None

    if not secret_matches(x_service_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid service key",
        )

    return x_service_key
