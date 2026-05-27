from sqlalchemy import JSON, Column
from sqlmodel import Field, Relationship

from chap_core.database.base_tables import DBModel
from chap_core.database.feature_tables import FeatureType, ModelFeatureLink
from chap_core.database.model_templates_and_config_tables import ModelTemplateMetaData
from chap_core.model_spec import PeriodType

# TODO: legacy, most here will be outdated, and should be moved elsewhere


class ModelSpecBase(ModelTemplateMetaData, DBModel):
    """Legacy base for the older `ModelSpec` table.

    Use inheritance here so that it's flat in the database.
    """

    name: str = Field(description="Canonical identifier of the model spec.")
    # supported_period_types: PeriodType = PeriodType.any
    source_url: str | None = Field(default=None, description="URL where the model's source lives.")
    supported_period_type: PeriodType = Field(
        default=PeriodType.any,
        description="Period granularity the model accepts (`month`, `week`, or `any`).",
    )

    # @field_validator("supported_period_type", mode="before")
    # def wrap_in_list(cls, v):
    #    if isinstance(v, list):
    #        return v
    #    return [v]


class ModelSpecRead(ModelSpecBase):
    """API read shape for a legacy `ModelSpec`.

    Carries the joined `covariates` / `target` references plus the
    archived / chapkit / configuration fields that the current frontend
    expects when listing available models.
    """

    id: int = Field(description="Primary key of the underlying `ModelSpec` row.")
    covariates: list[FeatureType] = Field(description="Covariate feature types this model supports.")
    target: FeatureType = Field(description="The feature type this model predicts.")
    archived: bool = Field(default=False, description="When True, the model is hidden from default pickers.")
    uses_chapkit: bool = Field(default=False, description="When True, the model is served by a chapkit REST endpoint.")
    user_option_values: dict | None = Field(default=None, description="Configured user-option values, if any.")
    additional_continuous_covariates: list[str] = Field(
        default=[],
        description="Extra continuous covariates passed beyond `covariates`.",
    )


class ModelSpec(ModelSpecBase, table=True):
    """
    ModelSpec is the DB class for a Configured Model.
    It is configured through the "configuration" field which is JSON
    """

    id: int | None = Field(primary_key=True, default=None, description="Primary key.")
    covariates: list[FeatureType] = Relationship(link_model=ModelFeatureLink)
    target_name: str = Field(
        foreign_key="featuretype.name", description="Canonical name of the predicted `FeatureType`."
    )
    target: FeatureType = Relationship()
    configuration: dict | None = Field(
        sa_column=Column(JSON),
        description="JSON blob holding the model's configured options.",
    )
