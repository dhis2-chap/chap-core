from typing import Optional, List

from pydantic_geojson import FeatureModel

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
    element_id: Optional[str]


class Observation(ObservationBase, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    dataset_id: int = Field(foreign_key="dataset.id")
    dataset: 'DataSet' = Relationship(back_populates="observations")


class DataSetBase(DBModel):
    name: str
    geojson: Optional[
        str] = None  # Optional[FeatureCollectionModel] = Field(default=None, sa_type=AutoString) #fix from https://github.com/fastapi/sqlmodel/discussions/730#discussioncomment-7952622
    type: Optional[str] = None


class DataSet(DataSetBase, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    observations: List[Observation] = Relationship(back_populates="dataset")


class DataSetWithObservations(DataSetBase):
    id: int
    observations: List[ObservationBase]
