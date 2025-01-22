from typing import Optional, List

from sqlalchemy import create_engine, Column, JSON
from sqlmodel import Field, Relationship, Session, select

from chap_core.database.base_tables import PeriodID, DBModel


class BackTestBase(DBModel):
    dataset_id: int = Field(foreign_key="dataset.id")
    model_id: str


class BackTest(BackTestBase, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    forecasts: List['BackTestForecast'] = Relationship(back_populates="backtest")
    metrics: List['BackTestMetric'] = Relationship(back_populates="backtest")


class ForecastBase(DBModel):
    period: PeriodID
    org_unit: str


class PredictionBase(DBModel):
    dataset_id: int = Field(foreign_key="dataset.id")
    estimator_id: str
    n_periods: int


class Prediction(PredictionBase, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    forecasts: List['PredictionForecast'] = Relationship(back_populates="prediction")


class ForecastRead(ForecastBase):
    values: List[float] = Field(default_factory=list, sa_column=Column(JSON))


class PredictionRead(PredictionBase):
    id: int
    forecasts: List[ForecastRead]


class PredictionForecast(ForecastBase, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    prediction_id: int = Field(foreign_key="prediction.id")
    prediction: 'Prediction' = Relationship(back_populates="forecasts")
    values: List[float] = Field(default_factory=list, sa_column=Column(JSON))


class BackTestForecast(ForecastBase, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    backtest_id: int = Field(foreign_key="backtest.id")
    last_train_period: PeriodID
    last_seen_period: PeriodID
    backtest: BackTest = Relationship(back_populates="forecasts")  # TODO: maybe remove this
    values: List[float] = Field(default_factory=list, sa_column=Column(JSON))


class BackTestMetric(DBModel, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    backtest_id: int = Field(foreign_key="backtest.id")
    metric_id: str
    period: PeriodID
    last_train_period: PeriodID
    last_seen_period: PeriodID
    value: float
    backtest: BackTest = Relationship(back_populates="metrics")


# merge request json/csv -
# change crud to v2 ?
# maybe have geometry in table
# direct predictions endpoint
# maybe add list of models to evaluate
# maybe add id to health data
# Maybe add a way to get the health data
# Hide session objects

# TODO:
# Maybe version the database ala flyway
# Alembic, maybe check out south
# Discuss metadata storing for showing informally to the user
# MAybe do it when doing landcover
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
