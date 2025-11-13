"""
todo: comment this file, make it clear which classes are central and being used
"""

import datetime
from typing import Optional, List, Dict

import numpy as np
from sqlalchemy import Column, JSON
from sqlmodel import Field, Relationship

from chap_core.database.base_tables import PeriodID, DBModel
from chap_core.database.dataset_tables import DataSet, DataSetInfo
from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB, ModelTemplateDB, ModelConfiguration


class BackTestBase(DBModel):
    dataset_id: int = Field(foreign_key="dataset.id")
    model_id: str
    name: Optional[str] = None
    created: Optional[datetime.datetime] = None
    model_template_version: Optional[str] = (
        None  # This is the version of the model template in the moment the backtest was created (version at model template object can change later)
    )


class DataSetMeta(DataSetInfo):
    id: int
    # created: datetime.datetime
    # covariates: List[str]


class _BackTestRead(BackTestBase):
    id: int
    org_units: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    split_periods: List[PeriodID] = Field(default_factory=list, sa_column=Column(JSON))


class BackTest(_BackTestRead, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    dataset: DataSet = Relationship()
    forecasts: List["BackTestForecast"] = Relationship(back_populates="backtest", cascade_delete=True)
    metrics: List["BackTestMetric"] = Relationship(back_populates="backtest", cascade_delete=True)
    aggregate_metrics: Dict[str, float] = Field(default_factory=dict, sa_column=Column(JSON))
    model_db_id: int = Field(foreign_key="configuredmodeldb.id")
    configured_model: Optional["ConfiguredModelDB"] = Relationship()


class ConfiguredModelRead(ModelConfiguration, DBModel):
    name: str
    id: int
    model_template: ModelTemplateDB


OldBackTestRead = _BackTestRead


class BackTestRead(_BackTestRead):
    dataset: DataSetMeta
    aggregate_metrics: Dict[str, float]
    configured_model: Optional[ConfiguredModelRead]


class ForecastBase(DBModel):
    period: PeriodID
    org_unit: str
    values: List[float] = Field(default_factory=list, sa_type=JSON)

    def get_quantiles(self, quantiles: List[float]) -> np.ndarray:
        return np.quantile(self.values, quantiles).astype(float)


class ForecastRead(ForecastBase): ...


class PredictionBase(DBModel):
    dataset_id: int = Field(foreign_key="dataset.id")
    model_id: str
    n_periods: int
    name: str
    created: datetime.datetime
    meta_data: dict = Field(default_factory=dict, sa_column=Column(JSON))
    org_units: List[str] = Field(default_factory=list, sa_column=Column(JSON))


class Prediction(PredictionBase, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    forecasts: List["PredictionSamplesEntry"] = Relationship(back_populates="prediction", cascade_delete=True)
    dataset: DataSet = Relationship()
    model_db_id: int = Field(foreign_key="configuredmodeldb.id")
    configured_model: Optional["ConfiguredModelDB"] = Relationship()


class PredictionInfo(PredictionBase):
    id: int
    configured_model: Optional[ConfiguredModelDB]
    dataset: DataSetMeta


# PredictionInfo = PredictionBase.get_read_class()


class PredictionRead(PredictionInfo):
    forecasts: List[ForecastRead]


class PredictionSamplesEntry(ForecastBase, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    prediction_id: int = Field(foreign_key="prediction.id")
    prediction: "Prediction" = Relationship(back_populates="forecasts")


class BackTestForecast(ForecastBase, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    backtest_id: int = Field(foreign_key="backtest.id")
    last_train_period: PeriodID
    last_seen_period: PeriodID
    backtest: BackTest = Relationship(back_populates="forecasts")


class BackTestMetric(DBModel, table=True):
    """
    This class has been used when computing metrics per location/time_point/split_point adhoc
    in database.py. This id depcrecated and not used in the new metric system.
    Can be removed when no references left to this class.
    """

    id: Optional[int] = Field(primary_key=True, default=None)
    backtest_id: int = Field(foreign_key="backtest.id")
    metric_id: str
    period: PeriodID  # Should this be optional and be null for aggregate metrics?
    org_unit: str  # Should this be optional and be null for aggregate metrics?
    last_train_period: PeriodID
    last_seen_period: PeriodID
    value: float
    backtest: BackTest = Relationship(back_populates="metrics")


# def test():
#     engine = create_engine("sqlite://")
#     DBModel.metadata.create_all(engine)
#     with Session(engine) as session:
#         backtest = BackTest(dataset_id="dataset_id", model_id="model_id")
#         forecast = BackTestForecast(
#             period="202101",
#             org_unity="RegionA",
#             last_train_period="202012",
#             last_seen_period="202012",
#             values=[1.0, 2.0, 3.0],
#         )
#         metric = BackTestMetric(
#             metric_id="metric_id", period="202101", last_train_period="202012", last_seen_period="202012", value=0.5
#         )
#         backtest.forecasts.append(forecast)
#         backtest.metrics.append(metric)
#         session.add(backtest)
#         session.commit()
#         print(session.exec(select(BackTestForecast)).all())
