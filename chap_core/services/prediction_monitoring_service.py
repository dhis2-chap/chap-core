"""Monitoring of a PredictionSetup: score each backtest split and each live prediction.

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
from chap_core.assessment.flat_representations import FlatForecasts, FlatObserved, horizon_diff
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
from chap_core.time_period import TimePeriod

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


def _origin_periods(forecasts: pd.DataFrame) -> pd.Series:
    """The last period each forecast row's run had data for: its target period minus its horizon."""
    origins = {
        (period, horizon): (TimePeriod.parse(period) - horizon * TimePeriod.parse(period).time_delta).id
        for period, horizon in forecasts[["time_period", "horizon_distance"]].drop_duplicates().itertuples(index=False)
    }
    return pd.Series(
        [origins[key] for key in zip(forecasts.time_period, forecasts.horizon_distance, strict=True)],
        index=forecasts.index,
    )


def metric_over_time(
    metric: Metric, observations: pd.DataFrame, forecasts: pd.DataFrame, source: MonitoringSource
) -> list[MonitoringPoint]:
    """One point per run (backtest split or prediction), keyed by the period it predicted from.

    A run is scored on the forecasts that have an observed value so far, pooled over org
    units and horizons; the running value pools every run up to and including it.
    """
    if observations.empty or forecasts.empty:
        return []
    observed = FlatObserved.validate(observations[observations.disease_cases.notna()])
    observed_keys = set(zip(observed.location, observed.time_period, strict=True))
    is_scored = pd.Series(
        [key in observed_keys for key in zip(forecasts.location, forecasts.time_period, strict=True)],
        index=forecasts.index,
    )
    scored = forecasts[is_scored].assign(origin=lambda df: _origin_periods(df))
    points = []
    for origin in sorted(scored.origin.unique()):
        run = scored[scored.origin == origin]
        value = metric.get_global_metric(observed, FlatForecasts.validate(run.drop(columns="origin")))
        running = metric.get_global_metric(
            observed, FlatForecasts.validate(scored[scored.origin <= origin].drop(columns="origin"))
        )
        points.append(
            MonitoringPoint(
                period=origin,
                source=source,
                value=float(value.metric.iloc[0]),
                running_value=float(running.metric.iloc[0]),
                n_observed=len(run[["location", "time_period", "horizon_distance"]].drop_duplicates()),
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
