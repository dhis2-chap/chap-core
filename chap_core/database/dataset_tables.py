from datetime import datetime
from typing import Optional, List, Annotated, Any

from pydantic_geojson import FeatureModel
from sqlalchemy import JSON, Column, TypeDecorator, String
from sqlalchemy.types import UserDefinedType
from pydantic import field_validator
import json

from sqlmodel import Field, Relationship

from chap_core.database.base_tables import DBModel, PeriodID
from pydantic_geojson import (
    FeatureCollectionModel as _FeatureCollectionModel,
)


class FeatureCollectionModel(_FeatureCollectionModel):
    features: list[FeatureModel]


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
            return [
                item.model_dump() if hasattr(item, 'model_dump') else item
                for item in value
            ]
        return value

    def process_result_value(self, value, dialect):
        """Convert JSON from database back to list of Pydantic models"""
        if value is None:
            return []
        if isinstance(value, list):
            return [
                self.pydantic_model_class(**item) if isinstance(item, dict) else item
                for item in value
            ]
        return value


class ObservationBase(DBModel):
    period: PeriodID
    org_unit: str
    value: Optional[float]
    feature_name: Optional[str]


class Observation(ObservationBase, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    dataset_id: int = Field(foreign_key="dataset.id")
    dataset: "DataSet" = Relationship(back_populates="observations")

class DataSource(DBModel):
    covariate: str
    data_element_id: str

class DataSetBase(DBModel):
    name: str
    type: Optional[str] = None
    geojson: Optional[str] = None
    data_sources: List[DataSource] = Field(default_factory=list, sa_column=Column(PydanticListType(DataSource)))
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
