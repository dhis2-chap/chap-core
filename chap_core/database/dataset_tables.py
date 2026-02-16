from datetime import datetime

from pydantic_geojson import (
    FeatureCollectionModel as _FeatureCollectionModel,
)
from pydantic_geojson import FeatureModel
from sqlalchemy import JSON, Column, TypeDecorator
from sqlmodel import Field, Relationship

from chap_core.database.base_tables import DBModel, PeriodID


class FeatureCollectionModel(_FeatureCollectionModel):
    features: list[FeatureModel]  # type: ignore[assignment]


class PydanticListType(TypeDecorator):
    """Custom SQLAlchemy type that automatically serializes/deserializes Pydantic model lists"""

    impl = JSON
    cache_ok = True

    def __init__(self, pydantic_model_class):
        self.pydantic_model_class = pydantic_model_class
        super().__init__()

    def process_bind_param(self, value, dialect):
        """Convert Python list of Pydantic models to JSON for database storage"""
        if value is None:
            return None
        if isinstance(value, list):
            return [item.model_dump() if hasattr(item, "model_dump") else item for item in value]
        return value

    def process_result_value(self, value, dialect):
        """Convert JSON from database back to list of Pydantic models"""
        if value is None:
            return []
        if isinstance(value, list):
            return [self.pydantic_model_class(**item) if isinstance(item, dict) else item for item in value]
        return value


class ObservationBase(DBModel):
    period: PeriodID
    org_unit: str
    value: float | None
    feature_name: str | None


class Observation(ObservationBase, table=True):
    id: int | None = Field(primary_key=True, default=None)
    dataset_id: int = Field(foreign_key="dataset.id")
    dataset: "DataSet" = Relationship(back_populates="observations")


class DataSource(DBModel):
    covariate: str
    data_element_id: str


class DataSetCreateInfo(DBModel):
    name: str = Field(description="Name of dataset")

    data_sources: list[DataSource] | None = Field(
        default_factory=list,
        sa_column=Column(PydanticListType(DataSource)),
        description="A mapping of covariate names to data element IDs from which to source the data",
    )
    type: str | None = Field(None, description="Purpose of dataset, e.g., 'forecasting' or 'backtesting'")


class DataSetInfo(DataSetCreateInfo):
    id: int | None = Field(primary_key=True, default=None)
    covariates: list["str"] = Field(default_factory=list, sa_column=Column(JSON))
    first_period: PeriodID | None = Field(default=None)
    last_period: PeriodID | None = Field(default=None)
    org_units: list["str"] | None = Field(default_factory=list, sa_column=Column(JSON))
    created: datetime | None = None

    period_type: str | None = None


class DataSetBase(DataSetInfo):
    geojson: str | None = None


class DataSet(DataSetBase, table=True):
    observations: list[Observation] = Relationship(back_populates="dataset", cascade_delete=True)


class DataSetWithObservations(DataSetBase):
    id: int
    observations: list[ObservationBase]
    created: datetime | None
