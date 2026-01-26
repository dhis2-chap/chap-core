"""
Orchestrator for managing chapkit service registration and discovery.

This module provides the Orchestrator class which handles:
- Service registration with automatic ULID generation
- Keepalive ping mechanism with TTL-based expiration
- Service discovery and listing
- Service deregistration

Services are stored in Redis with automatic expiration.
"""

import json
from datetime import datetime, timezone
from typing import Any

from redis import Redis
from ulid import ULID

from chap_core.rest_api.services.schemas import (
    PingResponse,
    RegistrationPayload,
    RegistrationResponse,
    ServiceDetail,
    ServiceListResponse,
)

DEFAULT_TTL_SECONDS = 30
KEY_PREFIX = "service:"


class ServiceNotFoundError(Exception):
    """Raised when a requested service is not found in the registry."""

    pass


class Orchestrator:
    """
    Manages chapkit service registration and discovery using Redis.

    Services register with the orchestrator and must send periodic pings
    to maintain their registration. Services that fail to ping within
    the TTL window are automatically expired by Redis.
    """

    def __init__(self, redis_client: Redis, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        """
        Initialize the Orchestrator.

        Args:
            redis_client: Redis client for storing service registrations.
            ttl_seconds: Time-to-live for service registrations in seconds.
                Services must ping within this interval to stay registered.
        """
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds

    def _make_key(self, service_id: str) -> str:
        """Create a Redis key for the given service ID."""
        return f"{KEY_PREFIX}{service_id}"

    def _now_iso(self) -> str:
        """Get current UTC timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()

    def _compute_expires_at(self) -> str:
        """Compute expiration timestamp based on current time and TTL."""
        from datetime import timedelta

        return (datetime.now(timezone.utc) + timedelta(seconds=self.ttl_seconds)).isoformat()

    def register(self, payload: RegistrationPayload) -> RegistrationResponse:
        """
        Register a new service with the orchestrator.

        Args:
            payload: Registration payload containing service URL and info.

        Returns:
            RegistrationResponse with the assigned service ID and ping URL.
        """
        service_id = str(ULID())
        now = self._now_iso()
        expires_at = self._compute_expires_at()

        service_data = {
            "id": service_id,
            "url": payload.url,
            "info": payload.info.model_dump(mode="json"),
            "registered_at": now,
            "last_updated": now,
            "last_ping_at": now,
            "expires_at": expires_at,
        }

        key = self._make_key(service_id)
        self.redis.setex(key, self.ttl_seconds, json.dumps(service_data))

        return RegistrationResponse(
            id=service_id,
            status="registered",
            service_url=payload.url,
            message="Service registered successfully",
            ttl_seconds=self.ttl_seconds,
            ping_url=f"/v2/services/{service_id}/$ping",
        )

    def ping(self, service_id: str) -> PingResponse:
        """
        Send a keepalive ping for a registered service.

        Updates the service's last_ping_at timestamp and resets the TTL.

        Args:
            service_id: The ULID of the service to ping.

        Returns:
            PingResponse with updated timestamps.

        Raises:
            ServiceNotFoundError: If the service is not registered.
        """
        key = self._make_key(service_id)
        data = self.redis.get(key)

        if data is None:
            raise ServiceNotFoundError(f"Service {service_id} not found")

        service_data: dict[str, Any] = json.loads(data)
        now = self._now_iso()
        expires_at = self._compute_expires_at()

        service_data["last_ping_at"] = now
        service_data["last_updated"] = now
        service_data["expires_at"] = expires_at

        self.redis.setex(key, self.ttl_seconds, json.dumps(service_data))

        return PingResponse(
            id=service_id,
            status="alive",
            last_ping_at=now,
            expires_at=expires_at,
        )

    def deregister(self, service_id: str) -> bool:
        """
        Deregister a service from the orchestrator.

        Args:
            service_id: The ULID of the service to deregister.

        Returns:
            True if the service was successfully deregistered.

        Raises:
            ServiceNotFoundError: If the service is not registered.
        """
        key = self._make_key(service_id)
        deleted = self.redis.delete(key)
        if deleted == 0:
            raise ServiceNotFoundError(f"Service {service_id} not found")
        return True

    def get(self, service_id: str) -> ServiceDetail:
        """
        Get details of a specific registered service.

        Args:
            service_id: The ULID of the service to retrieve.

        Returns:
            ServiceDetail containing the service's registration info.

        Raises:
            ServiceNotFoundError: If the service is not registered.
        """
        key = self._make_key(service_id)
        data = self.redis.get(key)

        if data is None:
            raise ServiceNotFoundError(f"Service {service_id} not found")

        service_data: dict[str, Any] = json.loads(data)
        return ServiceDetail(**service_data)

    def get_all(self) -> ServiceListResponse:
        """
        List all registered services.

        Returns:
            ServiceListResponse containing all currently registered services.
        """
        pattern = f"{KEY_PREFIX}*"
        keys = list(self.redis.scan_iter(pattern))

        services: list[ServiceDetail] = []
        for key in keys:
            data = self.redis.get(key)
            if data is not None:
                service_data: dict[str, Any] = json.loads(data)
                services.append(ServiceDetail(**service_data))

        return ServiceListResponse(count=len(services), services=services)
