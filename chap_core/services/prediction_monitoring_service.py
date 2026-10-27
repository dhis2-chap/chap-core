"""Monitoring of a PredictionSetup: compare each backtest split and each live prediction with a baseline model.

Live predictions are scored against the observed cases stored in
`PredictionSetupObservation`, which every run of the setup updates, so a forecast made
by one run is scored once a later run brings in the observed value for its period.
Backtest forecasts are scored against the backtest's own dataset. All org units and
horizons are pooled, matching how a backtest's aggregate metrics are computed.

The baseline is the installation's default baseline model. Its backtest on the setup's
backtest specification is the reference for the backtest splits, and the forecast it
made alongside each live prediction is the reference for that prediction. A metric's
comparison operator turns the two scores into one value, positive meaning better than
the baseline.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pandas as pd
from sqlmodel import col, select

from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.flat_representations import FlatForecasts, FlatObserved, horizon_diff
from chap_core.assessment.metrics import compare_aggregated_metrics, get_comparison_description, get_metric
from chap_core.database.database import SessionWrapper
from chap_core.database.settings import SettingsManager
from chap_core.database.tables import (
    Backtest,
    MonitoringPoint,
    MonitoringSource,
    Prediction,
    PredictionSetup,
    PredictionSetupMonitoring,
    PredictionSetupObservation,
    SpecificationBaseline,
    SpecificationBaselineStatus,
)
from chap_core.services.prediction_setup_service import PredictionSetupNotFoundError
from chap_core.time_period import TimePeriod

if TYPE_CHECKING:
    import pandera.pandas as pa
    from sqlmodel import Session

    from chap_core.assessment.metrics.base import Metric
    from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB


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


def _forecast_keys(forecasts: pd.DataFrame) -> pd.Series:
    return pd.Series(
        list(zip(forecasts.location, forecasts.time_period, forecasts.horizon_distance, strict=True)),
        index=forecasts.index,
    )


def _score(metric: Metric, observed: pa.typing.DataFrame[FlatObserved], forecasts: pd.DataFrame) -> float:
    return float(metric.get_global_metric(observed, FlatForecasts.validate(forecasts)).metric.iloc[0])


def _compare(metric: Metric, score: float, baseline_score: float) -> float | None:
    value = metric.compare(score, baseline_score)
    assert value is not None, f"Metric {metric.spec.metric_id!r} has no comparison operator"
    return None if math.isnan(value) else value


def comparison_over_time(
    metric: Metric,
    observations: pd.DataFrame,
    forecasts: pd.DataFrame,
    baseline_forecasts: pd.DataFrame,
    source: MonitoringSource,
) -> list[MonitoringPoint]:
    """One point per run (backtest split or prediction), keyed by the period it predicted from.

    A run is scored on the forecasts that both it and the baseline made and that have an
    observed value so far, pooled over org units and horizons, and the score is compared
    with the baseline's on the same forecasts. The running value compares every run up to
    and including it.
    """
    if observations.empty or forecasts.empty or baseline_forecasts.empty:
        return []
    observed = FlatObserved.validate(observations[observations.disease_cases.notna()])
    observed_locations_periods = set(zip(observed.location, observed.time_period, strict=True))
    scored_keys = {
        key
        for key in set(_forecast_keys(forecasts)) & set(_forecast_keys(baseline_forecasts))
        if key[:2] in observed_locations_periods
    }
    if not scored_keys:
        return []
    model, baseline = (
        frame[_forecast_keys(frame).isin(scored_keys)].assign(origin=lambda df: _origin_periods(df))
        for frame in (forecasts, baseline_forecasts)
    )
    points = []
    for origin in sorted(model.origin.unique()):
        scores = [
            (
                _score(metric, observed, frame[frame.origin == origin].drop(columns="origin")),
                _score(metric, observed, frame[frame.origin <= origin].drop(columns="origin")),
            )
            for frame in (model, baseline)
        ]
        (value, running), (baseline_value, baseline_running) = scores
        run = model[model.origin == origin]
        points.append(
            MonitoringPoint(
                period=origin,
                source=source,
                value=_compare(metric, value, baseline_value),
                running_value=_compare(metric, running, baseline_running),
                n_observed=len(run[["location", "time_period", "horizon_distance"]].drop_duplicates()),
            )
        )
    return points


def _baseline_backtest(session: Session, backtest: Backtest, baseline: ConfiguredModelDB) -> Backtest | None:
    """The baseline's backtest on the same specification, once it is done."""
    return session.exec(
        select(Backtest)
        .join(SpecificationBaseline, col(SpecificationBaseline.backtest_id) == Backtest.id)
        .where(
            SpecificationBaseline.specification_id == backtest.specification_id,
            SpecificationBaseline.configured_model_id == baseline.id,
            SpecificationBaseline.status == SpecificationBaselineStatus.done,
        )
    ).first()


def get_prediction_setup_monitoring(session: Session, setup_id: int, metric_id: str) -> PredictionSetupMonitoring:
    """Compare a setup's backtest and live predictions with the default baseline model over time for one metric.

    Raises:
        PredictionSetupNotFoundError: setup does not exist.
        InvalidMetricError: metric is unknown, has no comparison, or is not applicable to the observed data.
    """
    setup = session.get(PredictionSetup, setup_id)
    if setup is None:
        raise PredictionSetupNotFoundError(f"PredictionSetup {setup_id} not found")
    metric_cls = get_metric(metric_id)
    if metric_cls is None:
        raise InvalidMetricError(f"Unknown metric '{metric_id}'")
    if metric_cls.spec.comparison_op is None:
        raise InvalidMetricError(f"Metric '{metric_id}' cannot be compared with a baseline")
    metric = metric_cls()
    baseline_name = SettingsManager(session).get().default_baseline_model
    baseline = SessionWrapper(session=session).get_configured_model_by_id_or_name(baseline_name)

    evaluation = Evaluation.from_backtest(setup.backtest).to_flat()
    if not evaluation.observations.empty and not metric.is_applicable(evaluation.observations):
        raise InvalidMetricError(f"Metric '{metric_id}' cannot be computed for this setup's data")
    baseline_backtest = _baseline_backtest(session, setup.backtest, baseline)
    evaluation_points: list[MonitoringPoint] = []
    evaluation_value = None
    if baseline_backtest is not None:
        evaluation_points = comparison_over_time(
            metric,
            evaluation.observations,
            evaluation.forecasts,
            Evaluation.from_backtest(baseline_backtest).to_flat().forecasts,
            MonitoringSource.evaluation,
        )
        evaluation_value = compare_aggregated_metrics(
            setup.backtest.aggregate_metrics, baseline_backtest.aggregate_metrics
        ).get(metric_id)

    baseline_predictions = [
        baseline_prediction
        for prediction in setup.predictions
        for baseline_prediction in prediction.baseline_predictions
        if baseline_prediction.model_id == baseline_name
    ]
    prediction_points = comparison_over_time(
        metric,
        flat_setup_observations(setup.observations),
        flat_prediction_forecasts(setup.predictions),
        flat_prediction_forecasts(baseline_predictions),
        MonitoringSource.prediction,
    )
    return PredictionSetupMonitoring(
        metric_id=metric_id,
        baseline_model=baseline_name,
        comparison_description=get_comparison_description(metric_cls.spec.comparison_op),
        evaluation_value=evaluation_value,
        points=evaluation_points + prediction_points,
    )
