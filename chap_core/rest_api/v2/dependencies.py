import os
from functools import lru_cache

from fastapi import Header, HTTPException, status

from chap_core.rest_api.services.orchestrator import Orchestrator
from chap_core.util import load_redis

SERVICE_KEY_HEADER = "X-Service-Key"
SERVICE_KEY_ENV_VAR = "SERVICEKIT_REGISTRATION_KEY"


@lru_cache
def get_redis():
    return load_redis(db=3)


def get_orchestrator():
    return Orchestrator(redis_client=get_redis())


def verify_service_key(
    x_service_key: str | None = Header(default=None, alias=SERVICE_KEY_HEADER),
) -> str | None:
    """
    Verify the service registration API key.

    If SERVICEKIT_REGISTRATION_KEY is not configured, authentication is skipped.

    Raises:
        HTTPException 422: If key is configured but header is missing
        HTTPException 401: If the provided key doesn't match
    """
    expected_key = os.getenv(SERVICE_KEY_ENV_VAR)

    # If no key configured on server, skip authentication
    if not expected_key:
        return None

    # Key is configured, so header is required
    if not x_service_key:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="X-Service-Key header is required",
        )

    if x_service_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service key",
        )

    return x_service_key
