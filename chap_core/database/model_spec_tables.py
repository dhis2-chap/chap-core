from sqlalchemy import JSON, Column
from sqlmodel import Field, Relationship

from chap_core.database.base_tables import DBModel
from chap_core.database.feature_tables import FeatureType, ModelFeatureLink
from chap_core.database.model_templates_and_config_tables import ModelTemplateMetaData
from chap_core.model_spec import PeriodType

# TODO: legacy, most here will be outdated, and should be moved elsewhere


class ModelSpecBase(ModelTemplateMetaData, DBModel):
    """
    Use inheritance here so that it's flat in the database.
    """

    name: str
    # supported_period_types: PeriodType = PeriodType.any
    source_url: str | None = None
    supported_period_type: PeriodType = PeriodType.any  # ] = [PeriodType.month, PeriodType.week]

    # @field_validator("supported_period_type", mode="before")
    # def wrap_in_list(cls, v):
    #    if isinstance(v, list):
    #        return v
    #    return [v]


class ModelSpecRead(ModelSpecBase):
    id: int
    covariates: list[FeatureType]
    target: FeatureType
    archived: bool = False
    uses_chapkit: bool = False
    user_option_values: dict | None = None
    additional_continuous_covariates: list[str] = []


class ModelSpec(ModelSpecBase, table=True):
    """
    ModelSpec is the DB class for a Configured Model.
    It is configured through the "configuration" field which is JSON
    """

    id: int | None = Field(primary_key=True, default=None)
    covariates: list[FeatureType] = Relationship(link_model=ModelFeatureLink)
    target_name: str = Field(foreign_key="featuretype.name")
    target: FeatureType = Relationship()
    configuration: dict | None = Field(sa_column=Column(JSON))
