"""Stored defaults shared by the benchmark leaderboard and API backtest requests."""

import logging
from typing import Self

from pydantic import Field, ValidationError, field_validator, model_validator

from chap_core.api_types import BacktestParams
from chap_core.assessment.metrics import available_metrics
from chap_core.database.base_tables import DBModel
from chap_core.database.manager import DbManager
from chap_core.database.model_templates_and_config_tables import ModelTemplateRole
from chap_core.database.tables import BackendSettings
from chap_core.models.builtin import get_builtin_models

logger = logging.getLogger(__name__)

DEFAULT_METRIC_ORDER = [
    "crps_log1p",
    "winkler_score_25_75_log1p",
    "mae",
    "rmse",
    "mape",
    "coverage_10_90",
    "coverage_25_75",
    "crps",
    "crps_norm",
    "ratio_above_truth",
    "winkler_score_10_90",
    "winkler_score_10_90_log1p",
    "winkler_score_25_75",
]
DEFAULT_BASELINE_MODEL = "persistence"


def is_builtin_baseline(name: str) -> bool:
    model = get_builtin_models().get(name)
    return model is not None and model.spec.role == ModelTemplateRole.baseline


class Settings(DBModel):
    """Installation-wide defaults. PUT replaces the complete settings record."""

    metric_sort_order: list[str] = Field(
        description="Metric ids in display and ranking order. Unlisted metrics are appended in their default order."
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
        if not is_builtin_baseline(value):
            raise ValueError("The default model must be a built-in baseline.")
        return value

    @model_validator(mode="after")
    def complete_metric_order(self) -> Self:
        default_order = DEFAULT_METRIC_ORDER + [m for m in available_metrics if m not in DEFAULT_METRIC_ORDER]
        self.metric_sort_order = self.metric_sort_order + [
            metric_id for metric_id in default_order if metric_id not in self.metric_sort_order
        ]
        self.selected_metrics = [
            metric_id for metric_id in self.metric_sort_order if metric_id in self.selected_metrics
        ]
        return self

    def backtest_params(self, period_type: str | None) -> BacktestParams:
        if period_type == "week":
            return self.weekly_backtest_params
        if period_type == "month":
            return self.monthly_backtest_params
        logger.warning("No stored backtest defaults for period type %r; using schema defaults", period_type)
        return BacktestParams()


def default_settings() -> Settings:
    return Settings(
        metric_sort_order=DEFAULT_METRIC_ORDER,
        selected_metrics=DEFAULT_METRIC_ORDER[:6],
        default_baseline_model=DEFAULT_BASELINE_MODEL,
        weekly_backtest_params=BacktestParams(n_periods=12, stride=4),
        monthly_backtest_params=BacktestParams(),
    )


class SettingsManager(DbManager[BackendSettings]):
    """Stores only the settings an admin changed, so later changes to the code defaults still apply."""

    model = BackendSettings

    def get(self) -> Settings:
        """Code defaults with the stored overrides on top, dropping metrics and baselines that no longer exist.

        Falls back to the defaults if the overrides are otherwise invalid, for example after
        a weather provider is removed, so stale settings never break the API.
        """
        record = self.find_by_id(1)
        values = default_settings().model_dump() | (record.values if record else {})
        for key in ("metric_sort_order", "selected_metrics"):
            values[key] = [metric_id for metric_id in values[key] if metric_id in available_metrics]
        if not is_builtin_baseline(values["default_baseline_model"]):
            values["default_baseline_model"] = DEFAULT_BASELINE_MODEL
        try:
            return Settings.model_validate(values)
        except ValidationError:
            logger.warning("Stored backend settings are invalid; using defaults", exc_info=True)
            return default_settings()

    def update(self, settings: Settings) -> Settings:
        defaults = default_settings().model_dump()
        record = self.find_by_id(1) or BackendSettings()
        record.values = {key: value for key, value in settings.model_dump().items() if value != defaults[key]}
        self.save(record)
        return settings
