"""
Pydantic schemas for the chapkit service registration API.

These schemas define the request and response models for service
registration, discovery, and keepalive operations.

Note: The MLServiceInfo-related schemas (AssessedStatus, PeriodType, ModelMetadata,
ServiceInfo, MLServiceInfo) are duplicated from chapkit. In the future, these will
be replaced with imports from a shared chapkit data types package.
"""

import re
from enum import StrEnum

from pydantic import BaseModel, EmailStr, Field, HttpUrl, field_validator

SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")


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

    author: str | None = Field(default=None, description="Person or team that authored the model.")
    author_note: str | None = Field(
        default=None, description="Free-form note from the author (caveats, intended use, ...)."
    )
    author_assessed_status: AssessedStatus | None = Field(
        default=None, description="Author-declared maturity rating (gray/red/orange/yellow/green)."
    )
    contact_email: EmailStr | None = Field(default=None, description="Contact email for the model author / maintainer.")
    organization: str | None = Field(default=None, description="Affiliated organisation, if any.")
    organization_logo_url: HttpUrl | None = Field(
        default=None, description="URL of an organisation logo to render next to the model."
    )
    citation_info: str | None = Field(
        default=None, description="How to cite the model in publications (DOI, BibTeX, ...)."
    )
    repository_url: HttpUrl | None = Field(default=None, description="URL to the model's source code repository.")
    documentation_url: HttpUrl | None = Field(default=None, description="URL to the model's external documentation.")


class ServiceInfo(BaseModel):
    """Base service information metadata."""

    id: str = Field(description="Unique service identifier (slug format)")
    display_name: str = Field(description="Human-friendly service name shown to operators.")
    version: str = Field(default="1.0.0", description="Service version string.")
    description: str | None = Field(default=None, description="Short paragraph describing what the service does.")

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate service ID follows slug format."""
        if not SLUG_PATTERN.match(v):
            raise ValueError(
                "Service ID must be slug format: lowercase letters, numbers, "
                "and hyphens (e.g., 'my-service', 'chap-ewars')"
            )
        return v


class MLServiceInfo(ServiceInfo):
    """ML service information extending base ServiceInfo with model-specific fields."""

    model_metadata: ModelMetadata = Field(
        description="Author / documentation metadata for the model the service hosts."
    )
    period_type: PeriodType = Field(description="Period granularity the model accepts (`weekly` or `monthly`).")
    min_prediction_periods: int = Field(
        default=0, description="Minimum forecast horizon (in periods) the model supports."
    )
    max_prediction_periods: int = Field(
        default=100, description="Maximum forecast horizon (in periods) the model supports."
    )
    allow_free_additional_continuous_covariates: bool = Field(
        default=False,
        description="When True, callers can attach extra continuous covariates beyond `required_covariates`.",
    )
    required_covariates: list[str] = Field(
        default_factory=list,
        description="Covariate names the model must be given to run.",
    )
    requires_geo: bool = Field(
        default=False, description="When True, the model needs a GeoJSON polygon set for spatial features."
    )


class RegistrationRequest(BaseModel):
    """Request body for registering a new chapkit service."""

    url: str = Field(description="Base URL of the chapkit service")
    info: MLServiceInfo = Field(description="MLServiceInfo metadata from chapkit")


class RegistrationResponse(BaseModel):
    """Response returned after successful service registration."""

    id: str = Field(description="Unique service identifier (slug)")
    status: str = Field(description="Registration status, always 'registered'")
    service_url: str = Field(description="The registered service URL")
    message: str = Field(description="Human-readable status message")
    ttl_seconds: int = Field(description="Time-to-live in seconds before expiration")
    ping_url: str = Field(description="URL to use for keepalive pings")


class ServiceDetail(BaseModel):
    """Detailed information about a registered service."""

    id: str = Field(description="Unique service identifier (slug)")
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
