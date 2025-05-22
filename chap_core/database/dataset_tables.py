from datetime import datetime
from typing import Optional, List

from pydantic_geojson import FeatureModel
from sqlalchemy import JSON, Column

from sqlmodel import Field, Relationship

from chap_core.database.base_tables import DBModel, PeriodID
from pydantic_geojson import (
    FeatureCollectionModel as _FeatureCollectionModel,
)


class FeatureCollectionModel(_FeatureCollectionModel):
    features: list[FeatureModel]


class ObservationBase(DBModel):
    period: PeriodID
    org_unit: str
    value: Optional[float]
    feature_name: Optional[str]


class Observation(ObservationBase, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    dataset_id: int = Field(foreign_key="dataset.id")
    dataset: "DataSet" = Relationship(back_populates="observations")


class DataSetBase(DBModel):
    name: str
    type: Optional[str] = None
    geojson: Optional[str] = None
    # Optional[FeatureCollectionModel] = Field(default=None, sa_type=AutoString) #fix from https://github.com/fastapi/sqlmodel/discussions/730#discussioncomment-7952622


class DataSet(DataSetBase, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    observations: List[Observation] = Relationship(back_populates="dataset", cascade_delete=True)
    covariates: List["str"] = Field(default_factory=list, sa_column=Column(JSON))
    created: Optional[datetime] = None


class DataSetWithObservations(DataSetBase):
    id: int
    observations: List[ObservationBase]
    created: Optional[datetime]
