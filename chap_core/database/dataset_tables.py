from typing import Optional, List

from sqlmodel import Field, Relationship

from chap_core.database.base_tables import DBModel, PeriodID


class ObservationBase(DBModel):
    period: PeriodID
    org_unit: str
    value: Optional[float]
    element_id: str


class Observation(ObservationBase, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    dataset_id: int = Field(foreign_key="dataset.id")
    dataset: 'DataSet' = Relationship(back_populates="observations")


class DataSetBase(DBModel):
    name: str
    geojson: Optional[str] = Field(default=None)
    type: Optional[str] = None


class DataSet(DataSetBase, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    observations: List[Observation] = Relationship(back_populates="dataset")


class DataSetWithObservations(DataSetBase):
    id: int
    observations: List[ObservationBase]
