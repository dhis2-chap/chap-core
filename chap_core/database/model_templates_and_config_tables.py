import logging
from enum import Enum

import jsonschema
from pydantic import ConfigDict
from sqlalchemy import JSON, Column
from sqlmodel import Field, Relationship, SQLModel

from chap_core.database.base_tables import DBModel
from chap_core.model_spec import PeriodType

logger = logging.getLogger(__name__)


class AuthorAssessedStatus(Enum):
    """Author-declared maturity rating of a model template.

    Surfaced to operators in model pickers so users can tell experimental
    work apart from validated work. The colour scheme mirrors a traffic
    light from least to most trustworthy.
    """

    gray = "gray"  # Gray: Not intended for use, or deprecated/meant for legacy use only.
    red = "red"  # Red: Highly experimental prototype - not at all validated and only meant for early experimentation
    orange = "orange"  # Orange: Has seen promise on limited data, needs manual configuration and careful evaluation
    yellow = "yellow"  # Yellow: Ready for more rigorous testing
    green = "green"  # Green: Validated, ready for use


class ModelTemplateMetaData(SQLModel):
    """Human-facing metadata for a model template — what the model is, who wrote it, how to cite it."""

    display_name: str = Field(
        default="No Display Name yet", description="Human-friendly name shown in model pickers and plot titles."
    )
    description: str = Field(
        default="No Description yet", description="Short paragraph explaining what the model does."
    )
    author_note: str = Field(
        default="No Author note yet", description="Free-form note from the author (e.g. caveats, intended use cases)."
    )
    author_assessed_status: AuthorAssessedStatus = Field(
        default=AuthorAssessedStatus("red"),
        description="Author-declared maturity of the model (gray/red/orange/yellow/green).",
    )
    author: str = Field(default="Unknown Author", description="Person or team that authored the model.")
    organization: str | None = Field(default=None, description="Affiliated organisation, if any.")
    organization_logo_url: str | None = Field(
        default=None, description="URL of an organisation logo to render next to the model."
    )
    contact_email: str | None = Field(default=None, description="Contact email for the model author / maintainer.")
    citation_info: str | None = Field(
        default=None, description="How to cite the model in publications (e.g. DOI, BibTeX)."
    )
    documentation_url: str | None = Field(default=None, description="URL to the model's external documentation.")


class ModelTemplateInformation(SQLModel):
    """Technical capabilities of a model template — what inputs it expects and what it can produce."""

    supported_period_type: PeriodType = Field(
        default=PeriodType.any, description="Period granularity the template supports (`month`, `week`, or `any`)."
    )
    user_options: dict | None = Field(
        default_factory=dict,
        sa_column=Column(JSON),
        description="JSON-schema-like dict describing the user-configurable options the template exposes.",
    )
    hpo_search_space: dict | None = Field(
        default=None,
        sa_column=Column(JSON),
        description="Search space used by HPO when training this template in `hpo` mode.",
    )
    required_covariates: list[str] = Field(
        default_factory=list,
        sa_column=Column(JSON),
        description="Covariate names the template must be given to run.",
    )
    min_prediction_length: int | None = Field(
        default=None, description="Minimum forecast horizon (in periods) the template supports."
    )
    max_prediction_length: int | None = Field(
        default=None, description="Maximum forecast horizon (in periods) the template supports."
    )
    target: str = Field(default="disease_cases", description="Name of the variable the model predicts.")
    allow_free_additional_continuous_covariates: bool = Field(
        default=False,
        description="When True, callers can attach extra continuous covariates beyond `required_covariates`.",
    )
    requires_geo: bool = Field(
        default=False, description="When True, the template needs a GeoJSON polygon set for spatial features."
    )


class ModelTemplateDB(DBModel, ModelTemplateMetaData, ModelTemplateInformation, table=True):
    """Persisted model-template row. Flat composition of metadata + capability mixins."""

    name: str = Field(unique=True, description="Canonical unique identifier of the template.")
    id: int | None = Field(primary_key=True, default=None, description="Primary key.")
    source_url: str | None = Field(
        default=None, description="URL where the template's source lives (e.g. a GitHub repo)."
    )
    configured_models: list["ConfiguredModelDB"] = Relationship(back_populates="model_template", cascade_delete=True)
    version: str | None = Field(default=None, description="Template version string, typically a git tag or commit sha.")
    archived: bool = Field(
        default=False, description="When True, the template is hidden from default pickers but still resolvable."
    )
    uses_chapkit: bool = Field(
        default=False,
        description="When True, the template is served by a chapkit REST endpoint rather than an MLproject directory.",
    )


class ModelConfiguration(SQLModel):
    """A specific choice of user-option values + extra covariates layered on top of a model template."""

    model_config = ConfigDict(extra="forbid")  # type: ignore[assignment]

    user_option_values: dict | None = Field(
        sa_column=Column(JSON),
        default_factory=dict,
        description="Values for the user-options declared by the parent `ModelTemplateDB.user_options` schema.",
    )
    additional_continuous_covariates: list[str] = Field(
        default_factory=list,
        sa_column=Column(JSON),
        description="Extra continuous covariates to pass to the model beyond the template's required set.",
    )


class ConfiguredModelDB(ModelConfiguration, DBModel, table=True):
    """Persisted configured-model row — a `ModelTemplateDB` together with a specific configuration."""

    #  unique constraint on name
    # model_config = ConfigDict(protected_namespaces=())
    name: str = Field(
        unique=True,
        description="Canonical unique identifier; conventionally `<template_name>` or `<template_name>:<config_stub>`.",
    )
    id: int | None = Field(primary_key=True, default=None, description="Primary key.")
    model_template_id: int = Field(
        foreign_key="modeltemplatedb.id",
        ondelete="CASCADE",
        description="Foreign key to the parent `ModelTemplateDB`.",
    )
    model_template: ModelTemplateDB = Relationship(back_populates="configured_models")
    archived: bool = Field(default=False, description="When True, the configured model is hidden from default pickers.")
    uses_chapkit: bool = Field(
        default=False, description="Inherited from the template; True for chapkit-hosted models."
    )

    @property
    def display_name(self) -> str:
        """Derived display name stitched from the template and (optionally) a configuration stub.

        Configured models whose name contains ``:`` were created as
        ``<template_name>:<configuration_name>`` (see ``SessionWrapper.add_configured_model``);
        default configurations reuse their template's name verbatim.
        """
        template_display_name = self.model_template.display_name
        if ":" not in self.name:
            return template_display_name
        configuration_stub = self.name.rsplit(":", 1)[-1]
        configuration_display_name = configuration_stub.replace("_", " ").capitalize()
        return f"{template_display_name} [{configuration_display_name}]"

    @classmethod
    def _validate_model_configuration(cls, user_options, user_option_values):
        logger.debug("Validating model configuration")
        logger.debug(f"User options keys: {list(user_options.keys()) if user_options else []}")
        logger.debug(f"User option values keys: {list(user_option_values.keys()) if user_option_values else []}")
        schema = {
            "type": "object",
            "properties": user_options,
            "required": list({key for key, value in user_options.items() if "default" not in value}),
            "additionalProperties": False,
        }
        jsonschema.validate(instance=user_option_values, schema=schema)

    # @model_validator(mode='after')
    def validate_user_options(self, model):
        try:
            self._validate_model_configuration(model.model_template.user_options, model.user_option_values)
        except jsonschema.ValidationError as e:
            logger.error(f"Validation error in model configuration: {e}")
            raise ValueError(f"Invalid user options: {e.message}") from e
        return model
