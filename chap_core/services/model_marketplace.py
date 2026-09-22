"""Resolve reviewed chapkit image pins from the CHAP model marketplace."""

import os
from dataclasses import dataclass
from typing import Literal

import httpx
import yaml
from pydantic import BaseModel, Field

DEFAULT_REGISTRY_URL = "https://raw.githubusercontent.com/dhis2-chap/model-marketplace/main"


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


class MarketplaceModel(BaseModel):
    schema_version: Literal[2]
    id: str
    service_id: str
    kind: Literal["model", "template"]
    source: ModelSource
    channels: dict[str, str]
    versions: list[ModelVersion]


@dataclass(frozen=True)
class ModelPin:
    image: str
    version: str


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
    return ModelPin(image=f"{entry.source.image}:{version.image_tag}", version=version.version)
