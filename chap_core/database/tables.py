"""
todo: comment this file, make it clear which classes are central and being used
"""

import datetime
from typing import Optional

import numpy as np
from sqlalchemy import JSON, Column
from sqlmodel import Field, Relationship

from chap_core.database.base_tables import DBModel, PeriodID
from chap_core.database.dataset_tables import DataSet, DataSetInfo, DataSource, PydanticListType
from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB, ModelConfiguration, ModelTemplateDB


class BacktestBase(DBModel):
    dataset_id: int = Field(foreign_key="dataset.id")
    model_id: str
    name: str | None = None
    created: datetime.datetime | None = None
    model_template_version: str | None = (
        None  # This is the version of the model template in the moment the backtest was created (version at model template object can change later)
    )


class DataSetMeta(DataSetInfo):
    id: int
    # created: datetime.datetime
    # covariates: List[str]


class _BacktestRead(BacktestBase):
    id: int
    org_units: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    split_periods: list[PeriodID] = Field(default_factory=list, sa_column=Column(JSON))


class Backtest(_BacktestRead, table=True):
    id: int | None = Field(primary_key=True, default=None)  # type: ignore[assignment]
    dataset: DataSet = Relationship()
    forecasts: list["BacktestForecast"] = Relationship(back_populates="backtest", cascade_delete=True)
    metrics: list["BacktestMetric"] = Relationship(back_populates="backtest", cascade_delete=True)
    aggregate_metrics: dict[str, float] = Field(default_factory=dict, sa_column=Column(JSON))
    model_db_id: int = Field(foreign_key="configuredmodeldb.id")
    configured_model: Optional["ConfiguredModelDB"] = Relationship()


class ConfiguredModelRead(ModelConfiguration, DBModel):
    name: str
    id: int
    model_template: ModelTemplateDB


class ConfiguredModelWithDataSource(DBModel, table=True):
    id: int | None = Field(primary_key=True, default=None)
    name: str
    created: datetime.datetime | None = None
    configured_model_id: int = Field(foreign_key="configuredmodeldb.id")
    configured_model: Optional["ConfiguredModelDB"] = Relationship()
    start_period: PeriodID | None = None
    org_units: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    data_sources: list[DataSource] = Field(
        default_factory=list,
        sa_column=Column(PydanticListType(DataSource)),
    )
    period_type: str | None = None
    predictions: list["Prediction"] = Relationship(back_populates="configured_model_with_data_source")


class ConfiguredModelWithDataSourceRead(DBModel):
    id: int
    name: str
    created: datetime.datetime | None
    configured_model: ConfiguredModelRead | None
    start_period: PeriodID | None
    org_units: list[str]
    data_sources: list[DataSource]
    period_type: str | None


OldBacktestRead = _BacktestRead


class BacktestRead(_BacktestRead):
    dataset: DataSetMeta
    aggregate_metrics: dict[str, float]
    configured_model: ConfiguredModelRead | None


class ForecastBase(DBModel):
    period: PeriodID
    org_unit: str
    values: list[float] = Field(default_factory=list, sa_type=JSON)

    def get_quantiles(self, quantiles: list[float]) -> np.ndarray:
        return np.quantile(self.values, quantiles).astype(float)


class ForecastRead(ForecastBase): ...


class PredictionBase(DBModel):
    dataset_id: int = Field(foreign_key="dataset.id")
    model_id: str
    n_periods: int
    name: str
    created: datetime.datetime
    meta_data: dict = Field(default_factory=dict, sa_column=Column(JSON))
    org_units: list[str] = Field(default_factory=list, sa_column=Column(JSON))


class Prediction(PredictionBase, table=True):
    id: int | None = Field(primary_key=True, default=None)
    forecasts: list["PredictionSamplesEntry"] = Relationship(back_populates="prediction", cascade_delete=True)
    dataset: DataSet = Relationship()
    model_db_id: int = Field(foreign_key="configuredmodeldb.id")
    configured_model: Optional["ConfiguredModelDB"] = Relationship()
    configured_model_with_data_source_id: int | None = Field(
        default=None, foreign_key="configuredmodelwithdatasource.id", nullable=True
    )
    configured_model_with_data_source: Optional["ConfiguredModelWithDataSource"] = Relationship(
        back_populates="predictions"
    )


class PredictionInfo(PredictionBase):
    id: int
    configured_model: ConfiguredModelDB | None
    dataset: DataSetMeta
    configured_model_with_data_source: ConfiguredModelWithDataSourceRead | None = None


# PredictionInfo = PredictionBase.get_read_class()


class PredictionRead(PredictionInfo):
    forecasts: list[ForecastRead]


class ConfiguredModelWithDataSourceReadWithPredictions(ConfiguredModelWithDataSourceRead):
    predictions: list[PredictionInfo] = []


class PredictionSamplesEntry(ForecastBase, table=True):
    id: int | None = Field(primary_key=True, default=None)
    prediction_id: int = Field(foreign_key="prediction.id")
    prediction: "Prediction" = Relationship(back_populates="forecasts")


class BacktestForecast(ForecastBase, table=True):
    id: int | None = Field(primary_key=True, default=None)
    backtest_id: int = Field(foreign_key="backtest.id")
    last_train_period: PeriodID
    last_seen_period: PeriodID
    backtest: Backtest = Relationship(back_populates="forecasts")


class BacktestMetric(DBModel, table=True):
    """
    This class has been used when computing metrics per location/time_point/split_point adhoc
    in database.py. This id depcrecated and not used in the new metric system.
    Can be removed when no references left to this class.
    """

    id: int | None = Field(primary_key=True, default=None)
    backtest_id: int = Field(foreign_key="backtest.id")
    metric_id: str
    period: PeriodID  # Should this be optional and be null for aggregate metrics?
    org_unit: str  # Should this be optional and be null for aggregate metrics?
    last_train_period: PeriodID
    last_seen_period: PeriodID
    value: float
    backtest: Backtest = Relationship(back_populates="metrics")


# def test():
#     engine = create_engine("sqlite://")
#     DBModel.metadata.create_all(engine)
#     with Session(engine) as session:
#         backtest = Backtest(dataset_id="dataset_id", model_id="model_id")
#         forecast = BacktestForecast(
#             period="202101",
#             org_unity="RegionA",
#             last_train_period="202012",
#             last_seen_period="202012",
#             values=[1.0, 2.0, 3.0],
#         )
#         metric = BacktestMetric(
#             metric_id="metric_id", period="202101", last_train_period="202012", last_seen_period="202012", value=0.5
#         )
#         backtest.forecasts.append(forecast)
#         backtest.metrics.append(metric)
#         session.add(backtest)
#         session.commit()
#         print(session.exec(select(BacktestForecast)).all())
