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
    """Shared fields for every backtest shape (DB row, create request, read view)."""

    dataset_id: int = Field(
        foreign_key="dataset.id", description="Foreign key to the `DataSet` the backtest evaluates against."
    )
    model_id: str = Field(description="Name of the configured model that was backtested.")
    name: str | None = Field(default=None, description="Optional human-friendly name for the backtest.")
    created: datetime.datetime | None = Field(
        default=None, description="Server-side timestamp when the backtest row was created."
    )
    model_template_version: str | None = Field(
        default=None,
        description="Snapshot of the parent template's version at backtest-creation time (may differ from current template version).",
    )


class DataSetMeta(DataSetInfo):
    """Slim dataset summary embedded inside other read views (e.g. `BacktestRead`)."""

    id: int = Field(description="Primary key of the dataset.")
    # created: datetime.datetime
    # covariates: List[str]


class _BacktestRead(BacktestBase):
    """Internal base class for `Backtest` / `BacktestRead` carrying the JSON-stored org_units + split_periods."""

    id: int = Field(description="Primary key of the backtest.")
    org_units: list[str] = Field(
        default_factory=list,
        sa_column=Column(JSON),
        description="Identifiers of every org unit the backtest scored predictions over.",
    )
    split_periods: list[PeriodID] = Field(
        default_factory=list,
        sa_column=Column(JSON),
        description="Periods at which the rolling backtest's train/test split was advanced.",
    )


class Backtest(_BacktestRead, table=True):
    """Persisted backtest row. Owns its forecasts, metrics, and (optionally) a `PredictionSetup`."""

    id: int | None = Field(primary_key=True, default=None, description="Primary key.")  # type: ignore[assignment]
    dataset: DataSet = Relationship()
    forecasts: list["BacktestForecast"] = Relationship(back_populates="backtest", cascade_delete=True)
    metrics: list["BacktestMetric"] = Relationship(back_populates="backtest", cascade_delete=True)
    aggregate_metrics: dict[str, float] = Field(
        default_factory=dict,
        sa_column=Column(JSON),
        description="Map of metric id to aggregated score across all splits / org units.",
    )
    model_db_id: int = Field(
        foreign_key="configuredmodeldb.id",
        description="Foreign key to the `ConfiguredModelDB` row used to run the backtest.",
    )
    configured_model: Optional["ConfiguredModelDB"] = Relationship()
    prediction_setup: Optional["PredictionSetup"] = Relationship(
        back_populates="backtest",
        sa_relationship_kwargs={"uselist": False},
        cascade_delete=True,
    )

    @property
    def prediction_setup_id(self) -> int | None:
        # Exposed on BacktestRead so the UI can answer "does this backtest have a setup?"
        # without a second round-trip. Requires `prediction_setup` to be eager-loaded by
        # the caller (selectinload(Backtest.prediction_setup)) to avoid a lazy-load fail
        # in detached-session contexts.
        return self.prediction_setup.id if self.prediction_setup is not None else None


class ConfiguredModelRead(ModelConfiguration, DBModel):
    """API read shape for a `ConfiguredModelDB` plus its parent template."""

    name: str = Field(description="Canonical name of the configured model.")
    id: int = Field(description="Primary key of the configured model.")
    model_template: ModelTemplateDB = Field(description="Parent template the configuration extends.")


class QuantileTarget(DBModel):
    """A `(quantile, data_element_id)` pair declaring where to push a quantile of the predictive distribution.

    Used inside `PredictionSetup.quantile_targets` to tell the scheduled run
    which DHIS2 data element receives which quantile (e.g. median → element
    A, 90th-percentile → element B).
    """

    quantile: str = Field(
        description="Quantile as a numeric string (`'0.5'` for median, `'0.9'` for the 90th percentile, ...)."
    )
    data_element_id: str = Field(description="External data element id the quantile value is pushed to.")


class PredictionSetup(DBModel, table=True):
    """Persisted prediction-setup row: a recurring forecast schedule attached to a backtest."""

    id: int | None = Field(primary_key=True, default=None, description="Primary key.")
    name: str = Field(description="Human-friendly name for the setup.")
    created: datetime.datetime | None = Field(
        default=None, description="Server-side timestamp when the setup was created."
    )
    backtest_id: int = Field(
        foreign_key="backtest.id",
        unique=True,
        description="Foreign key to the parent `Backtest` (one-to-one).",
    )
    backtest: "Backtest" = Relationship(back_populates="prediction_setup")
    configured_model_id: int = Field(
        foreign_key="configuredmodeldb.id",
        description="Foreign key to the configured model the setup will run.",
    )
    configured_model: "ConfiguredModelDB" = Relationship()
    start_period: PeriodID | None = Field(
        default=None, description="Period the recurring run starts forecasting from; `None` means run-time default."
    )
    org_units: list[str] = Field(
        default_factory=list,
        sa_column=Column(JSON),
        description="Identifiers of the org units the setup forecasts for.",
    )
    covariate_sources: list[DataSource] = Field(
        default_factory=list,
        sa_column=Column(PydanticListType(DataSource)),
        description="Mapping of covariate names to data element ids used to source the inputs.",
    )
    period_type: str | None = Field(
        default=None, description="Granularity of the forecast periods (`month`, `week`, ...)."
    )
    schedule_cron_expression: str | None = Field(
        default=None,
        description="Standard cron expression for when the setup runs; `None` means manual-only.",
    )
    schedule_enabled: bool = Field(
        default=False, description="When True, the scheduler executes the setup at every cron tick."
    )
    quantile_targets: list[QuantileTarget] = Field(
        default_factory=list,
        sa_column=Column(PydanticListType(QuantileTarget)),
        description="Where to push each quantile of the predictive distribution.",
    )
    predictions: list["Prediction"] = Relationship(back_populates="prediction_setup")


class PredictionSetupRead(DBModel):
    """API read shape for a `PredictionSetup`. Same fields as the DB row but with the configured-model joined."""

    id: int = Field(description="Primary key of the setup.")
    name: str = Field(description="Human-friendly name for the setup.")
    created: datetime.datetime | None = Field(description="Server-side timestamp when the setup was created.")
    backtest_id: int = Field(description="Foreign key to the parent `Backtest`.")
    configured_model: ConfiguredModelRead = Field(description="Configured model the setup will run.")
    start_period: PeriodID | None = Field(
        description="Period the recurring run starts forecasting from; `None` means run-time default."
    )
    org_units: list[str] = Field(description="Identifiers of the org units the setup forecasts for.")
    covariate_sources: list[DataSource] = Field(
        description="Mapping of covariate names to data element ids used to source the inputs."
    )
    period_type: str | None = Field(description="Granularity of the forecast periods (`month`, `week`, ...).")
    schedule_cron_expression: str | None = Field(
        description="Standard cron expression for when the setup runs; `None` means manual-only."
    )
    schedule_enabled: bool = Field(description="When True, the scheduler executes the setup at every cron tick.")
    quantile_targets: list[QuantileTarget] = Field(
        description="Where to push each quantile of the predictive distribution."
    )


OldBacktestRead = _BacktestRead


class BacktestRead(_BacktestRead):
    """API read shape for a `Backtest`. Same fields as the DB row plus the joined dataset / model / setup links."""

    dataset: DataSetMeta = Field(description="Slim dataset summary the backtest evaluated against.")
    aggregate_metrics: dict[str, float] = Field(
        description="Map of metric id to aggregated score across all splits / org units."
    )
    configured_model: ConfiguredModelRead | None = Field(
        description="Configured model used for the backtest, joined for convenience."
    )
    prediction_setup_id: int | None = Field(
        default=None, description="Id of the attached `PredictionSetup`, if one exists."
    )


class ForecastBase(DBModel):
    """Shared shape for a single (period, org_unit) forecast carrying its sample values."""

    period: PeriodID = Field(description="Period the forecast is for.")
    org_unit: str = Field(description="Identifier of the org unit the forecast is for.")
    values: list[float] = Field(
        default_factory=list,
        sa_type=JSON,
        description="Posterior sample values for this (period, org_unit). Quantiles are derived from these at read time.",
    )

    def get_quantiles(self, quantiles: list[float]) -> np.ndarray:
        return np.quantile(self.values, quantiles).astype(float)


class ForecastRead(ForecastBase):
    """API read shape for a forecast — currently identical to `ForecastBase`."""


class PredictionBase(DBModel):
    """Shared fields for every prediction shape (DB row, read view, ...)."""

    dataset_id: int = Field(
        foreign_key="dataset.id", description="Foreign key to the `DataSet` the prediction was run against."
    )
    model_id: str = Field(description="Name of the configured model that produced the prediction.")
    n_periods: int = Field(description="Number of periods the model was asked to forecast.")
    name: str = Field(description="Human-friendly name for the prediction run.")
    created: datetime.datetime = Field(description="Server-side timestamp when the prediction row was created.")
    meta_data: dict = Field(
        default_factory=dict,
        sa_column=Column(JSON),
        description="Free-form metadata bag attached to the prediction (e.g. provenance, scheduler context).",
    )
    org_units: list[str] = Field(
        default_factory=list,
        sa_column=Column(JSON),
        description="Identifiers of the org units the prediction was run for.",
    )


class Prediction(PredictionBase, table=True):
    """Persisted prediction row. Owns its per-period forecasts and optionally belongs to a `PredictionSetup`."""

    id: int | None = Field(primary_key=True, default=None, description="Primary key.")
    forecasts: list["PredictionSamplesEntry"] = Relationship(back_populates="prediction", cascade_delete=True)
    dataset: DataSet = Relationship()
    model_db_id: int = Field(
        foreign_key="configuredmodeldb.id",
        description="Foreign key to the `ConfiguredModelDB` row used to run the prediction.",
    )
    configured_model: Optional["ConfiguredModelDB"] = Relationship()
    prediction_setup_id: int | None = Field(
        default=None,
        foreign_key="predictionsetup.id",
        nullable=True,
        description="Foreign key to the `PredictionSetup` that triggered the run, if any.",
    )
    prediction_setup: Optional["PredictionSetup"] = Relationship(back_populates="predictions")


class PredictionInfo(PredictionBase):
    """Summary read view for a prediction — fields + joined dataset/model, no per-period forecasts."""

    id: int = Field(description="Primary key of the prediction.")
    configured_model: ConfiguredModelDB | None = Field(
        description="Configured model used for the prediction, joined for convenience."
    )
    dataset: DataSetMeta = Field(description="Slim dataset summary the prediction was run against.")


# PredictionInfo = PredictionBase.get_read_class()


class PredictionRead(PredictionInfo):
    """Full prediction read view: summary fields plus every per-period forecast."""

    forecasts: list[ForecastRead] = Field(description="Per-(period, org_unit) forecast rows for this prediction.")


class PredictionSetupReadWithPredictions(PredictionSetupRead):
    """`PredictionSetupRead` augmented with the list of predictions the setup has produced so far."""

    predictions: list[PredictionInfo] = Field(
        default=[],
        description="Predictions produced by this setup so far, summary view only.",
    )


class PredictionSamplesEntry(ForecastBase, table=True):
    """Persisted forecast row that belongs to a `Prediction` (not a backtest)."""

    id: int | None = Field(primary_key=True, default=None, description="Primary key.")
    prediction_id: int = Field(foreign_key="prediction.id", description="Foreign key to the owning `Prediction`.")
    prediction: "Prediction" = Relationship(back_populates="forecasts")


class BacktestForecast(ForecastBase, table=True):
    """Persisted forecast row that belongs to a `Backtest` — adds train/seen-period context."""

    id: int | None = Field(primary_key=True, default=None, description="Primary key.")
    backtest_id: int = Field(foreign_key="backtest.id", description="Foreign key to the owning `Backtest`.")
    last_train_period: PeriodID = Field(
        description="Most recent period included in the training window for this forecast."
    )
    last_seen_period: PeriodID = Field(
        description="Most recent period whose actuals were visible when this forecast was generated."
    )
    backtest: Backtest = Relationship(back_populates="forecasts")


class BacktestMetric(DBModel, table=True):
    """
    This class has been used when computing metrics per location/time_point/split_point adhoc
    in database.py. This id depcrecated and not used in the new metric system.
    Can be removed when no references left to this class.
    """

    id: int | None = Field(primary_key=True, default=None, description="Primary key.")
    backtest_id: int = Field(foreign_key="backtest.id", description="Foreign key to the owning `Backtest`.")
    metric_id: str = Field(description="Canonical metric identifier (e.g. `crps`, `mae`).")
    period: PeriodID = Field(
        description="Period this metric value applies to (currently always populated; aggregates handled elsewhere)."
    )
    # Should this be optional and be null for aggregate metrics?
    org_unit: str = Field(
        description="Org unit this metric value applies to (currently always populated; aggregates handled elsewhere)."
    )
    # Should this be optional and be null for aggregate metrics?
    last_train_period: PeriodID = Field(
        description="Most recent period included in the training window for this score."
    )
    last_seen_period: PeriodID = Field(
        description="Most recent period whose actuals were visible when this score was computed."
    )
    value: float = Field(description="Computed metric value.")
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
