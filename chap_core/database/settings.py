"""Stored defaults shared by the benchmark leaderboard and API backtest requests."""

from typing import Self

from pydantic import Field, field_validator, model_validator
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from chap_core.api_types import BacktestParams
from chap_core.assessment.metrics import available_metrics
from chap_core.database.base_tables import DBModel
from chap_core.database.model_templates_and_config_tables import ModelTemplateRole
from chap_core.database.tables import BackendSettings
from chap_core.models.builtin import get_builtin_model


class Settings(DBModel):
    """Installation-wide defaults. PUT replaces the complete settings record."""

    metric_sort_order: list[str] = Field(
        description="Metric ids in display and ranking order. Unlisted metrics follow in their seeded order."
    )
    selected_metrics: list[str] = Field(description="Metric ids shown by default, ordered by metricSortOrder.")
    default_baseline_model: str = Field(
        description="Built-in baseline name, independent of its version (for example 'persistence')."
    )
    weekly_backtest_params: BacktestParams = Field(description="Defaults for backtests of weekly data.")
    monthly_backtest_params: BacktestParams = Field(description="Defaults for backtests of monthly data.")

    @field_validator("metric_sort_order", "selected_metrics")
    @classmethod
    def validate_metrics(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Metric ids must be unique.")
        unknown = set(value) - available_metrics.keys()
        if unknown:
            raise ValueError(f"Unknown metrics: {', '.join(sorted(unknown))}")
        return value

    @field_validator("default_baseline_model")
    @classmethod
    def validate_baseline(cls, value: str) -> str:
        if get_builtin_model(value).spec.role != ModelTemplateRole.baseline:
            raise ValueError("The default model must be a built-in baseline.")
        return value

    @model_validator(mode="after")
    def order_selected_metrics(self) -> Self:
        if not set(self.selected_metrics) <= set(self.metric_sort_order):
            raise ValueError("Selected metrics must be included in metricSortOrder.")
        self.selected_metrics = [
            metric_id for metric_id in self.metric_sort_order if metric_id in self.selected_metrics
        ]
        return self

    def backtest_params(self, period_type: str | None) -> BacktestParams:
        if period_type == "week":
            return self.weekly_backtest_params
        if period_type == "month":
            return self.monthly_backtest_params
        return BacktestParams()


def default_metric_order() -> list[str]:
    return sorted(available_metrics, key=lambda metric_id: (available_metrics[metric_id].spec.sort_order, metric_id))


def default_settings() -> Settings:
    order = default_metric_order()
    return Settings(
        metric_sort_order=order,
        selected_metrics=[metric_id for metric_id in order if available_metrics[metric_id].spec.default_selected],
        default_baseline_model="persistence",
        weekly_backtest_params=BacktestParams(n_periods=12),
        monthly_backtest_params=BacktestParams(),
    )


def get_settings_record(session: Session) -> BackendSettings:
    """Seed once, including when two API workers start against a fresh database."""
    record = session.get(BackendSettings, 1)
    if record is None:
        record = BackendSettings(values=default_settings().model_dump())
        session.add(record)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            record = session.get(BackendSettings, 1)
            if record is None:
                raise
    return record


def get_settings(session: Session) -> Settings:
    return Settings.model_validate(get_settings_record(session).values)
