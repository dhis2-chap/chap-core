from typing import Any

from pydantic import BaseModel


class RegistrationPayload(BaseModel):
    url: str  # Base URL of the chapkit service
    info: dict[str, Any]  # ServiceInfo from chapkit


class RegistrationResponse(BaseModel):
    id: str  # ULID
    status: str  # "registered"
    service_url: str
    message: str
    ttl_seconds: int
    ping_url: str


class ServiceDetail(BaseModel):
    id: str
    url: str
    info: dict[str, Any]
    registered_at: str
    last_updated: str
    last_ping_at: str
    expires_at: str


class ServiceListResponse(BaseModel):
    count: int
    services: list[ServiceDetail]


class PingResponse(BaseModel):
    id: str
    status: str  # "alive"
    last_ping_at: str
    expires_at: str
