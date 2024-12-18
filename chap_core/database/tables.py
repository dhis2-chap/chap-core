from typing import Optional, List

from sqlalchemy import create_engine, Column, JSON
from sqlmodel import Field, SQLModel, Relationship, Session, select

PeriodID = str


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
    backtest: BackTest = Relationship(back_populates="forecasts") # TODO: maybe remove this


class BackTestMetric(SQLModel, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    backtest_id: int = Field(foreign_key="backtest.id")
    metric_id: str
    period_id: PeriodID
    last_train_period_id: PeriodID
    last_seen_period_id: PeriodID
    value: float
    backtest: BackTest = Relationship(back_populates="metrics")


class Observation(SQLModel, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    period_id: PeriodID
    region_id: str
    value: Optional[float]
    element_id: str
    dataset_id: int = Field(foreign_key="dataset.id")


class DataSet(SQLModel, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    name: str
    polygons: Optional[str] = Field(default=None)
    observations: List[Observation] = Relationship()


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
