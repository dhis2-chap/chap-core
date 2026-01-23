import os
from functools import lru_cache

from fastapi import Header, HTTPException, status

from chap_core.rest_api.services.orchestrator import Orchestrator
from chap_core.util import load_redis

SERVICE_KEY_HEADER = "X-Service-Key"
SERVICE_KEY_ENV_VAR = "CHAP_SERVICE_REGISTRATION_KEY"


@lru_cache
def get_redis():
    return load_redis(db=3)


def get_orchestrator():
    return Orchestrator(redis_client=get_redis())


def verify_service_key(x_service_key: str = Header(alias=SERVICE_KEY_HEADER)) -> str:
    """
    Verify the service registration API key.

    Raises:
        HTTPException 503: If CHAP_SERVICE_REGISTRATION_KEY is not configured
        HTTPException 401: If the provided key doesn't match
    """
    expected_key = os.getenv(SERVICE_KEY_ENV_VAR)

    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service registration is not configured",
        )

    if x_service_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service key",
        )

    return x_service_key
