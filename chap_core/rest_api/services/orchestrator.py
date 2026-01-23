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
    pass


class Orchestrator:
    def __init__(self, redis_client: Redis, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds

    def _make_key(self, service_id: str) -> str:
        return f"{KEY_PREFIX}{service_id}"

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _compute_expires_at(self) -> str:
        from datetime import timedelta

        return (datetime.now(timezone.utc) + timedelta(seconds=self.ttl_seconds)).isoformat()

    def register(self, payload: RegistrationPayload) -> RegistrationResponse:
        service_id = str(ULID())
        now = self._now_iso()
        expires_at = self._compute_expires_at()

        service_data = {
            "id": service_id,
            "url": payload.url,
            "info": payload.info,
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
        key = self._make_key(service_id)
        deleted = self.redis.delete(key)
        if deleted == 0:
            raise ServiceNotFoundError(f"Service {service_id} not found")
        return True

    def get(self, service_id: str) -> ServiceDetail:
        key = self._make_key(service_id)
        data = self.redis.get(key)

        if data is None:
            raise ServiceNotFoundError(f"Service {service_id} not found")

        service_data: dict[str, Any] = json.loads(data)
        return ServiceDetail(**service_data)

    def get_all(self) -> ServiceListResponse:
        pattern = f"{KEY_PREFIX}*"
        keys = list(self.redis.scan_iter(pattern))

        services: list[ServiceDetail] = []
        for key in keys:
            data = self.redis.get(key)
            if data is not None:
                service_data: dict[str, Any] = json.loads(data)
                services.append(ServiceDetail(**service_data))

        return ServiceListResponse(count=len(services), services=services)
