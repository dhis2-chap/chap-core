"""
Pydantic schemas for the chapkit service registration API.

These schemas define the request and response models for service
registration, discovery, and keepalive operations.

Note: The MLServiceInfo-related schemas (AssessedStatus, PeriodType, ModelMetadata,
ServiceInfo, MLServiceInfo) are duplicated from chapkit. In the future, these will
be replaced with imports from a shared chapkit data types package.
"""

from enum import StrEnum

from pydantic import BaseModel, EmailStr, Field, HttpUrl


class AssessedStatus(StrEnum):
    """Model assessment status levels."""

    gray = "gray"
    red = "red"
    orange = "orange"
    yellow = "yellow"
    green = "green"


class PeriodType(StrEnum):
    """Supported time period types for model predictions."""

    weekly = "weekly"
    monthly = "monthly"


class ModelMetadata(BaseModel):
    """Metadata about the ML model author and documentation."""

    author: str | None = None
    author_note: str | None = None
    author_assessed_status: AssessedStatus | None = None
    contact_email: EmailStr | None = None
    organization: str | None = None
    organization_logo_url: HttpUrl | None = None
    citation_info: str | None = None
    repository_url: HttpUrl | None = None
    documentation_url: HttpUrl | None = None


class ServiceInfo(BaseModel):
    """Base service information metadata."""

    display_name: str
    version: str = "1.0.0"
    summary: str | None = None
    description: str | None = None
    contact: dict[str, str] | None = None
    license_info: dict[str, str] | None = None


class MLServiceInfo(ServiceInfo):
    """ML service information extending base ServiceInfo with model-specific fields."""

    model_metadata: ModelMetadata
    period_type: PeriodType
    min_prediction_periods: int = 0
    max_prediction_periods: int = 100
    allow_free_additional_continuous_covariates: bool = False
    required_covariates: list[str] = Field(default_factory=list)
    requires_geo: bool = False


class RegistrationPayload(BaseModel):
    """Payload for registering a new chapkit service."""

    url: str = Field(description="Base URL of the chapkit service")
    info: MLServiceInfo = Field(description="MLServiceInfo metadata from chapkit")


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
    info: MLServiceInfo = Field(description="MLServiceInfo metadata from chapkit")
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
