from typing import List, Optional

from sqlalchemy import Column, JSON
from sqlmodel import Field

from chap_core.database.base_tables import DBModel
from chap_core.model_spec import PeriodType


class FeatureTypeBase(DBModel):
    display_name: str
    description: str


class FeatureTypeRead(FeatureTypeBase):
    name: str


class FeatureType(FeatureTypeBase, table=True):
    name: str = Field(str, primary_key=True)


class FeatureSource(DBModel, table=True):
    name: str = Field(primary_key=True)
    display_name: str
    feature_type: str = Field(foreign_key="featuretype.name")
    provider: str
    supported_period_types: List[PeriodType] = Field(default_factory=list, sa_column=Column(JSON))


class ModelFeatureLink(DBModel, table=True):
    model_id: Optional[int] = Field(default=None, foreign_key="modelspec.id", primary_key=True)
    feature_type: Optional[str] = Field(default=None, foreign_key="featuretype.name", primary_key=True)
