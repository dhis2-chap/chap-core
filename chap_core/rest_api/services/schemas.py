"""
Pydantic schemas for the chapkit service registration API.

These schemas define the request and response models for service
registration, discovery, and keepalive operations.
"""

from typing import Any

from pydantic import BaseModel, Field


class RegistrationPayload(BaseModel):
    """Payload for registering a new chapkit service."""

    url: str = Field(description="Base URL of the chapkit service")
    info: dict[str, Any] = Field(description="ServiceInfo metadata from chapkit")


class RegistrationResponse(BaseModel):
    """Response returned after successful service registration."""

    id: str = Field(description="Unique service identifier (ULID)")
    status: str = Field(description="Registration status, always 'registered'")
    service_url: str = Field(description="The registered service URL")
    message: str = Field(description="Human-readable status message")
    ttl_seconds: int = Field(description="Time-to-live in seconds before expiration")
    ping_url: str = Field(description="URL to use for keepalive pings")


class ServiceDetail(BaseModel):
    """Detailed information about a registered service."""

    id: str = Field(description="Unique service identifier (ULID)")
    url: str = Field(description="Base URL of the chapkit service")
    info: dict[str, Any] = Field(description="ServiceInfo metadata from chapkit")
    registered_at: str = Field(description="ISO timestamp when service was registered")
    last_updated: str = Field(description="ISO timestamp of last update")
    last_ping_at: str = Field(description="ISO timestamp of last keepalive ping")
    expires_at: str = Field(description="ISO timestamp when registration expires")


class ServiceListResponse(BaseModel):
    """Response containing a list of registered services."""

    count: int = Field(description="Total number of registered services")
    services: list[ServiceDetail] = Field(description="List of service details")


class PingResponse(BaseModel):
    """Response returned after successful keepalive ping."""

    id: str = Field(description="Service identifier that was pinged")
    status: str = Field(description="Service status, always 'alive' on success")
    last_ping_at: str = Field(description="ISO timestamp of this ping")
    expires_at: str = Field(description="New expiration timestamp after TTL reset")
