from typing import Optional, List

from sqlalchemy import create_engine, Column, JSON
from sqlmodel import Field, SQLModel, Relationship, Session, select

PeriodID = str


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


class BackTest(SQLModel, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    dataset_id: int = Field(foreign_key="dataset.id")
    estimator_id: str
    forecasts: List['BackTestForecast'] = Relationship(back_populates="backtest")
    metrics: List['BackTestMetric'] = Relationship(back_populates="backtest")


class BackTestForecast(SQLModel, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    backtest_id: int = Field(foreign_key="backtest.id")
    period_id: PeriodID
    region_id: str
    last_train_period_id: PeriodID
    last_seen_period_id: PeriodID
    values: List[float] = Field(default_factory=list, sa_column=Column(JSON))
    backtest: BackTest = Relationship(back_populates="forecasts")  # TODO: maybe remove this


class BackTestMetric(SQLModel, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    backtest_id: int = Field(foreign_key="backtest.id")
    metric_id: str
    period_id: PeriodID
    last_train_period_id: PeriodID
    last_seen_period_id: PeriodID
    value: float
    backtest: BackTest = Relationship(back_populates="metrics")


class ObservationBase(SQLModel):
    period_id: PeriodID
    region_id: str
    value: Optional[float]
    element_id: str


class Observation(ObservationBase, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    dataset_id: int = Field(foreign_key="dataset.id")
    dataset: 'DataSet' = Relationship(back_populates="observations")


class DataSetBase(SQLModel):
    name: str
    polygons: Optional[str] = Field(default=None)


class DataSet(DataSetBase, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    observations: List[Observation] = Relationship(back_populates="dataset")


class DataSetWithObservations(DataSetBase):
    id: int
    observations: List[ObservationBase]


class DebugEntry(SQLModel, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    timestamp: float


def test():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        backtest = BackTest(dataset_id="dataset_id", model_id="model_id")
        forecast = BackTestForecast(period_id="202101", region_id="RegionA", last_train_period_id="202012",
                                    last_seen_period_id="202012", values=[1.0, 2.0, 3.0])
        metric = BackTestMetric(metric_id="metric_id", period_id="202101", last_train_period_id="202012",
                                last_seen_period_id="202012", value=0.5)
        backtest.forecasts.append(forecast)
        backtest.metrics.append(metric)
        session.add(backtest)
        session.commit()
        print(session.exec(select(BackTestForecast)).all())
