"""Resolve reviewed chapkit image pins from the CHAP model marketplace."""

import os
from dataclasses import dataclass
from typing import Any, Literal

import httpx
import yaml
from pydantic import BaseModel, Field

DEFAULT_REGISTRY_URL = "https://raw.githubusercontent.com/dhis2-chap/model-marketplace/main"

# BaseConfig fields chapkit reserves. CHAP supplies them itself at run time, so they are not
# user options of a template and not option values of a configuration.
RESERVED_CONFIG_KEYS = ("prediction_periods", "additional_continuous_covariates")


def registry_url() -> str:
    """Base URL of the marketplace registry, overridable with CHAP_MARKETPLACE_URL."""
    return os.getenv("CHAP_MARKETPLACE_URL", DEFAULT_REGISTRY_URL).rstrip("/")


class Registry(BaseModel):
    schema_version: Literal[2]
    models: list[str]


class ModelVersion(BaseModel):
    version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}$")
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    image_tag: str = Field(pattern=r"^[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,127}$")
    status: str


class ModelSource(BaseModel):
    image: str = Field(pattern=r"^[a-z0-9][a-z0-9._/\-]*$")
    repository: str | None = None


class Attribution(BaseModel):
    author: str | None = None
    organization: str | None = None
    contact: str | None = None
    citation: str | None = None


class Compatibility(BaseModel):
    period_types: list[Literal["weekly", "monthly"]] = []
    min_prediction_periods: int | None = None
    max_prediction_periods: int | None = None
    requires_geo: bool = False


class Covariates(BaseModel):
    required: list[str] = []
    defaults: list[str] = []
    allow_free_additional: bool = False


class Configuration(BaseModel):
    description: str | None = None
    config: dict[str, Any]


class MarketplaceModel(BaseModel):
    schema_version: Literal[2]
    id: str
    service_id: str
    kind: Literal["model", "template"]
    display_name: str | None = None
    assessed_status: Literal["gray", "red", "orange", "yellow", "green"] | None = None
    summary: str | None = None
    source: ModelSource
    attribution: Attribution = Attribution()
    compatibility: Compatibility = Compatibility()
    covariates: Covariates = Covariates()
    channels: dict[str, str]
    versions: list[ModelVersion]
    configurations: dict[str, Configuration] = {}


@dataclass(frozen=True)
class ModelPin:
    image: str
    version: str
    commit: str
    entry: MarketplaceModel


def resolve_model(model: str, base_url: str | None = None) -> ModelPin:
    """Return only the registry's verified stable pin, never the latest channel."""
    base_url = (base_url or registry_url()).rstrip("/")
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        response = client.get(f"{base_url}/registry.yaml")
        response.raise_for_status()
        registry = Registry.model_validate(yaml.safe_load(response.text))
        model_file = f"models/{model}.yaml"
        if model_file not in registry.models:
            raise ValueError(f"Model '{model}' is not listed in the marketplace.")
        response = client.get(f"{base_url}/{model_file}")
        response.raise_for_status()
        entry = MarketplaceModel.model_validate(yaml.safe_load(response.text))

    if entry.id != model or entry.service_id != model.replace("_", "-"):
        raise ValueError(f"Marketplace identity does not match '{model}'.")
    if entry.kind != "model":
        raise ValueError(f"'{model}' is a template for model authors, not a forecasting model.")
    stable = entry.channels.get("stable")
    version = next((version for version in entry.versions if version.version == stable), None)
    if version is None or version.status != "verified":
        raise ValueError(f"Model '{model}' has no verified stable version.")
    # Only an immutable sha- tag naming this version's commit is accepted; tags such as
    # latest, main or v1 can be moved to different code after the version was verified.
    prefix = version.image_tag[4:] if version.image_tag.startswith("sha-") else ""
    if len(prefix) < 7 or not version.commit.startswith(prefix):
        raise ValueError(f"Model '{model}' has an invalid stable image pin; expected a sha-<commit> tag.")
    return ModelPin(
        image=f"{entry.source.image}:{version.image_tag}", version=version.version, commit=version.commit, entry=entry
    )


def service_url(service_id: str) -> str:
    """Where a marketplace service is reachable from CHAP inside the Compose network."""
    return f"http://marketplace-{service_id}:8000"


def model_template_request(pin: ModelPin) -> dict[str, Any]:
    """The model template a pin describes, as the body of ``POST /v1/crud/model-templates``.

    The template is named after the chapkit service id, which is what the service reports
    as its own id, so the live service registry and the stored template line up. The
    registry does not carry the template's user option schema; CHAP fills it in from the
    service the first time it is registered and reachable.
    """
    entry = pin.entry
    period_types = set(entry.compatibility.period_types)
    if period_types == {"monthly"}:
        period_type = "month"
    elif period_types == {"weekly"}:
        period_type = "week"
    else:
        period_type = "any"
    return {
        "name": entry.service_id,
        "version": pin.version,
        "source_digest": pin.commit,
        "source_url": service_url(entry.service_id),
        "uses_chapkit": True,
        "display_name": entry.display_name or entry.id,
        "description": entry.summary or "No Description",
        "author": entry.attribution.author or "Unknown Author",
        "organization": entry.attribution.organization,
        "contact_email": entry.attribution.contact,
        "citation_info": entry.attribution.citation,
        "author_assessed_status": entry.assessed_status or "red",
        "documentation_url": entry.source.repository,
        "supported_period_type": period_type,
        "required_covariates": entry.covariates.required,
        "allow_free_additional_continuous_covariates": entry.covariates.allow_free_additional,
        "requires_geo": entry.compatibility.requires_geo,
        "min_prediction_periods": entry.compatibility.min_prediction_periods,
        "max_prediction_periods": entry.compatibility.max_prediction_periods,
    }


def configured_model_requests(pin: ModelPin, model_template_id: int) -> list[dict[str, Any]]:
    """One ``POST /v1/crud/configured-models`` body per verified configuration of the pin.

    A registry ``config`` is the flat object the chapkit service accepts. Its reserved
    BaseConfig keys are split out: the covariate list is stored as such, and the horizon
    is dropped because CHAP sets it per run.
    """
    requests = []
    for name, configuration in pin.entry.configurations.items():
        config = dict(configuration.config)
        covariates = config.pop("additional_continuous_covariates", pin.entry.covariates.defaults)
        for key in RESERVED_CONFIG_KEYS:
            config.pop(key, None)
        requests.append(
            {
                "name": name,
                "model_template_id": model_template_id,
                "user_option_values": config,
                "additional_continuous_covariates": list(covariates),
            }
        )
    return requests
