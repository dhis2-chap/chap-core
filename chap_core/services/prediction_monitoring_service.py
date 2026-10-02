"""Monitoring of a PredictionSetup: score its backtest and its live predictions period by period.

Live predictions are scored against the observed cases stored in
`PredictionSetupObservation`, which every run of the setup updates, so a forecast made
by one run is scored once a later run brings in the observed value for its period.
Backtest forecasts are scored against the backtest's own dataset. All org units and
horizons are pooled, matching how a backtest's aggregate metrics are computed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.flat_representations import DataDimension, FlatForecasts, FlatObserved, horizon_diff
from chap_core.assessment.metrics import get_metric
from chap_core.database.tables import (
    MonitoringPoint,
    MonitoringSource,
    Prediction,
    PredictionSetup,
    PredictionSetupMonitoring,
    PredictionSetupObservation,
)
from chap_core.services.prediction_setup_service import PredictionSetupNotFoundError

if TYPE_CHECKING:
    from sqlmodel import Session

    from chap_core.assessment.metrics.base import Metric


class InvalidMetricError(Exception):
    """Raised when the metric is unknown or cannot be computed for the setup's data."""


def flat_prediction_forecasts(predictions: list[Prediction]) -> pd.DataFrame:
    """Flatten a setup's predictions into the forecasts frame the metrics registry takes.

    Horizon distance counts from each prediction's first forecast period. Where runs
    overlap, the newest prediction for a (location, period, horizon) wins.
    """
    forecast_samples: dict[tuple[str, str, int], list[float]] = {}
    for prediction in sorted(predictions, key=lambda p: p.created):
        if not prediction.forecasts:
            continue
        first_period = min(entry.period for entry in prediction.forecasts)
        for entry in prediction.forecasts:
            forecast_samples[(entry.org_unit, entry.period, horizon_diff(entry.period, first_period))] = entry.values
    return pd.DataFrame(
        [
            {"location": location, "time_period": period, "horizon_distance": horizon, "sample": i, "forecast": value}
            for (location, period, horizon), values in forecast_samples.items()
            for i, value in enumerate(values)
        ],
        columns=["location", "time_period", "horizon_distance", "sample", "forecast"],
    )


def flat_setup_observations(observations: list[PredictionSetupObservation]) -> pd.DataFrame:
    """Flatten a setup's stored observed cases into the observations frame the metrics registry takes."""
    return pd.DataFrame(
        [
            {"location": obs.org_unit, "time_period": obs.period, "disease_cases": obs.disease_cases}
            for obs in observations
        ],
        columns=["location", "time_period", "disease_cases"],
    )


def metric_over_time(
    metric: Metric, observations: pd.DataFrame, forecasts: pd.DataFrame, source: MonitoringSource
) -> list[MonitoringPoint]:
    """Metric value per scored period, plus the running aggregate over all periods up to it."""
    if observations.empty or forecasts.empty:
        return []
    observed = observations[observations.disease_cases.notna()]
    scored_periods = sorted(set(forecasts.time_period) & set(observed.time_period))
    if not scored_periods:
        return []
    per_period = metric.get_metric(
        FlatObserved.validate(observed), FlatForecasts.validate(forecasts), dimensions=(DataDimension.time_period,)
    )
    values = dict(zip(per_period.time_period, per_period.metric, strict=True))
    points = []
    for period in scored_periods:
        running = metric.get_global_metric(
            FlatObserved.validate(observed[observed.time_period <= period]),
            FlatForecasts.validate(forecasts[forecasts.time_period <= period]),
        )
        points.append(
            MonitoringPoint(
                period=period,
                source=source,
                value=float(values[period]),
                running_value=float(running.metric.iloc[0]),
            )
        )
    return points


def get_prediction_setup_monitoring(session: Session, setup_id: int, metric_id: str) -> PredictionSetupMonitoring:
    """Score a setup's backtest and live predictions over time for one metric.

    Raises:
        PredictionSetupNotFoundError: setup does not exist.
        InvalidMetricError: metric is unknown or not applicable to the observed data.
    """
    setup = session.get(PredictionSetup, setup_id)
    if setup is None:
        raise PredictionSetupNotFoundError(f"PredictionSetup {setup_id} not found")
    metric_cls = get_metric(metric_id)
    if metric_cls is None:
        raise InvalidMetricError(f"Unknown metric '{metric_id}'")
    metric = metric_cls()

    evaluation = Evaluation.from_backtest(setup.backtest).to_flat()
    if not evaluation.observations.empty and not metric.is_applicable(evaluation.observations):
        raise InvalidMetricError(f"Metric '{metric_id}' cannot be computed for this setup's data")
    evaluation_points = metric_over_time(
        metric, evaluation.observations, evaluation.forecasts, MonitoringSource.evaluation
    )
    prediction_points = metric_over_time(
        metric,
        flat_setup_observations(setup.observations),
        flat_prediction_forecasts(setup.predictions),
        MonitoringSource.prediction,
    )
    return PredictionSetupMonitoring(
        metric_id=metric_id,
        evaluation_value=setup.backtest.aggregate_metrics.get(metric_id),
        points=evaluation_points + prediction_points,
    )
