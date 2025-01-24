from typing import Optional, List

from sqlalchemy import JSON, Column

from chap_core.database.base_tables import DBModel
from chap_core.model_spec import PeriodType
from sqlmodel import Field, Relationship


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


# TODO: Move to db spec

class ModelSpecBase(DBModel):
    name: str
    supported_period_types: PeriodType = PeriodType.any
    description: str = "No Description yet"
    author: str = "Unknown Author"


class ModelSpecRead(ModelSpecBase):
    id: int
    features: List[FeatureTypeRead]
    target: FeatureTypeRead


class ModelSpec(ModelSpecBase, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    features: List[FeatureType] = Relationship(link_model=ModelFeatureLink)
    target_id: str = Field(foreign_key="featuretype.name")
    target: FeatureType = Relationship()


target_type = FeatureType(name='disease_cases',
                          display_name='Disease Cases',
                          description='Disease Cases')

seeded_feature_types = [
    FeatureType(name='rainfall',
                display_name='Rainfall',
                description='Rainfall'),
    FeatureType(name='mean_temperature',
                display_name='Mean Temperature',
                description='A measurement of mean temperature'),
    FeatureType(name='population',
                display_name='Population',
                description='Population'),
    target_type]
base_features = [seeded_feature_types[0], seeded_feature_types[1], seeded_feature_types[2]]

seeded_models = [
    ModelSpec(
        name="chap_ewars_monthly",
        parameters={},
        features=base_features,
        period=PeriodType.month,
        description="Monthly EWARS model",
        author="CHAP",
        github_link="https://github.com/sandvelab/chap_auto_ewars@58d56f86641f4c7b09bbb635afd61740deff0640",
        target=target_type
    ),
    ModelSpec(
        name="chap_ewars_weekly",
        parameters={},
        features=base_features,
        period=PeriodType.week,
        description="Weekly EWARS model",
        author="CHAP",
        github_link="https://github.com/sandvelab/chap_auto_ewars_weekly@737446a7accf61725d4fe0ffee009a682e7457f6",
        target=target_type
    ),
    ModelSpec(
        name="auto_regressive_weekly",
        parameters={},
        features=base_features,
        period=PeriodType.week,
        description="Weekly Deep Auto Regressive model",
        author="knutdrand",
        github_link="https://github.com/knutdrand/weekly_ar_model@36a537dac138af428a4167b2a89eac7dafd5d762",
        target=target_type
    ),
    ModelSpec(
        name="auto_regressive_monthly",
        parameters={},
        features=base_features,
        period=PeriodType.month,
        description="Monthly Deep Auto Regressive model",
        author="knutdrand",
        github_link="https://github.com/sandvelab/monthly_ar_model@cadd785872624b4bcd839a39f5e7020c25254c31",
        target=target_type
    ),
]
