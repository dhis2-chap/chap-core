from typing import Optional, List

from pydantic import ConfigDict
from sqlalchemy import create_engine, Column, JSON
from sqlmodel import Field, SQLModel, Relationship, Session, select
from pydantic.alias_generators import to_camel

# def to_camel(string: str) -> str:
#     parts = string.split('_')
#     return parts[0] + ''.join(word.capitalize() for word in parts[1:])


PeriodID = str


class DBModel(SQLModel):
    ''' Simple wrapper that uses camelCase for the field names for the rest-api'''
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True)


# class FeatureTypes(SQLModel, table=True):
#     name: str = Field(str, primary_key=True)
#     display_name: str
#     description: str
#
#
# class Model(SQLModel, table=True):
#     id: Optional[int] = Field(primary_key=True, default=None)
#     name: str
#     estimator_id: str
#     features: List[FeatureTypes] = Relationship(back_populates="model")
#
#
# class FeatureSources(SQLModel, table=True):
#     id: Optional[int] = Field(primary_key=True, default=None)
#     name: str
#     feature_type: str
#     url: str
#     #metadata: Optional[str] = Field(default=None)
#
#
# class LocalDataSource(SQLModel, table=True):
#     dhis2_id: str = Field(primary_key=True)
#     feature_types: List[FeatureTypes] = Relationship(back_populates="source")

class BackTestBase(DBModel):
    dataset_id: int = Field(foreign_key="dataset.id")
    model_id: str


class BackTest(BackTestBase, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    forecasts: List['BackTestForecast'] = Relationship(back_populates="backtest")
    metrics: List['BackTestMetric'] = Relationship(back_populates="backtest")


class BackTestForecast(DBModel, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    backtest_id: int = Field(foreign_key="backtest.id")
    period: PeriodID
    org_unit: str
    last_train_period: PeriodID
    last_seen_period: PeriodID
    values: List[float] = Field(default_factory=list, sa_column=Column(JSON))
    backtest: BackTest = Relationship(back_populates="forecasts")  # TODO: maybe remove this


class BackTestMetric(DBModel, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    backtest_id: int = Field(foreign_key="backtest.id")
    metric_id: str
    period: PeriodID
    last_train_period: PeriodID
    last_seen_period: PeriodID
    value: float
    backtest: BackTest = Relationship(back_populates="metrics")


class ObservationBase(DBModel):
    period: PeriodID
    org_unit: str
    value: Optional[float]
    element_id: str


# rename polygons to geojson
# merge request json/csv -
# change crud to v2 ?
# remove json endpoint
# add dataset pure
# maybe have geometry in table
# direct predictions endpoint
# add dataset type to dataset
# maybe add list of models to evaluate
# maybe add id to health data
# Maybe add a way to get the health data
# Hide session objects

class Observation(ObservationBase, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    dataset_id: int = Field(foreign_key="dataset.id")
    dataset: 'DataSet' = Relationship(back_populates="observations")


class DataSetBase(DBModel):
    name: str
    geojson: Optional[str] = Field(default=None)


class DataSet(DataSetBase, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    observations: List[Observation] = Relationship(back_populates="dataset")


class DataSetWithObservations(DataSetBase):
    id: int
    observations: List[ObservationBase]


class DebugEntry(DBModel, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    timestamp: float


def test():
    engine = create_engine("sqlite://")
    DBModel.metadata.create_all(engine)
    with Session(engine) as session:
        backtest = BackTest(dataset_id="dataset_id", model_id="model_id")
        forecast = BackTestForecast(period="202101", org_unity="RegionA", last_train_period="202012",
                                    last_seen_period="202012", values=[1.0, 2.0, 3.0])
        metric = BackTestMetric(metric_id="metric_id", period="202101", last_train_period="202012",
                                last_seen_period="202012", value=0.5)
        backtest.forecasts.append(forecast)
        backtest.metrics.append(metric)
        session.add(backtest)
        session.commit()
        print(session.exec(select(BackTestForecast)).all())
