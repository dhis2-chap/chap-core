"""
Outbreak detection metrics: sensitivity, specificity, and accuracy.

An "outbreak" is observed cases exceeding the seasonal baseline (mean + 2*std)
for a given location and season bucket — calendar month for monthly data,
epi-week for weekly. An "alert" is raised when more than
:data:`ALERT_SAMPLE_FRACTION` of the forecast samples exceed that threshold.

All three metrics score the same joined frame, built once by
:func:`outbreak_and_alert`. They differ only in which rows they keep and what
they compare, so the threshold and alert rules live in one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from chap_core.assessment.metrics import metric
from chap_core.assessment.metrics.base import (
    AggregationOp,
    Metric,
    MetricSpec,
)

# The outbreak metrics score forecasts against seasonal thresholds, computed by the
# threshold strategy module.
from chap_core.assessment.thresholds.seasonal import compute_seasonal_thresholds
from chap_core.time_period.vectorized import season_column

if TYPE_CHECKING:
    import pandera.pandas as pa

    from chap_core.assessment.flat_representations import FlatObserved

#: Fraction of forecast samples that must exceed the threshold for an alert to be raised.
ALERT_SAMPLE_FRACTION = 0.5

_OUTBREAK_COLUMNS = ["location", "time_period", "horizon_distance", "outbreak", "alert"]
_METRIC_DIMENSIONS = ["location", "time_period", "horizon_distance"]


def _season_of(thresholds: pd.DataFrame) -> str:
    """The season bucket a threshold frame is keyed on."""
    return "week" if "week" in thresholds.columns else "month"


def has_season_buckets(observations: pd.DataFrame) -> bool:
    """Whether the observations' time periods bucket into calendar months or epi-weeks."""
    if observations.empty:
        return False
    try:
        season_column(observations["time_period"])
    except (ValueError, KeyError, TypeError):
        return False
    return True


def outbreak_and_alert(
    historical_observations: pd.DataFrame | None,
    observations: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Label every scored cell as outbreak and alert against the seasonal threshold.

    Args:
        historical_observations: Observations the threshold is computed from, with
            columns ``[location, time_period, disease_cases]``. ``None`` yields an
            empty frame — the metrics are not applicable without it.
        observations: Observed cases to label, same columns.
        forecasts: Forecast samples, with columns
            ``[location, time_period, horizon_distance, sample, forecast]``.

    Returns:
        One row per ``(location, time_period, horizon_distance)`` that has both a
        computable threshold and a forecast, with columns
        ``[location, time_period, horizon_distance, outbreak, alert]``. ``outbreak``
        is 1.0 where observed cases exceed the threshold; ``alert`` is 1.0 where more
        than :data:`ALERT_SAMPLE_FRACTION` of the samples exceed it. Cells whose
        threshold cannot be computed (a single historical value gives an undefined
        standard deviation) are dropped.
    """
    empty = pd.DataFrame(columns=_OUTBREAK_COLUMNS)
    if historical_observations is None or historical_observations.empty:
        return empty
    if observations.empty or forecasts.empty:
        return empty

    thresholds = compute_seasonal_thresholds(historical_observations)
    if thresholds.empty:
        return empty
    season = _season_of(thresholds)

    obs = observations[["location", "time_period", "disease_cases"]].copy()
    obs_season, obs_buckets = season_column(obs["time_period"])
    if obs_season != season:
        # Historical and scored periods are at different frequencies; nothing to join on.
        return empty
    obs[season] = obs_buckets
    obs = obs.merge(thresholds, on=["location", season], how="left").dropna(subset=["threshold"])
    if obs.empty:
        return empty
    obs["outbreak"] = (obs["disease_cases"] > obs["threshold"]).astype(float)

    fc = forecasts.copy()
    fc[season] = season_column(fc["time_period"])[1]
    fc = fc.merge(thresholds, on=["location", season], how="left")
    fc["exceeds"] = (fc["forecast"] > fc["threshold"]).astype(float)
    alert = fc.groupby(_METRIC_DIMENSIONS, as_index=False)["exceeds"].mean()
    alert["alert"] = (alert["exceeds"] > ALERT_SAMPLE_FRACTION).astype(float)

    merged = obs[["location", "time_period", "outbreak"]].merge(
        alert[[*_METRIC_DIMENSIONS, "alert"]],
        on=["location", "time_period"],
        how="inner",
    )
    return pd.DataFrame(merged[_OUTBREAK_COLUMNS])


def _as_metric(rows: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    """Reshape a slice of the outbreak frame into the metric output contract."""
    result = rows[_METRIC_DIMENSIONS].copy()
    result["metric"] = values
    return result


class _OutbreakMetric(Metric):
    """Shared applicability rule for metrics scored against a seasonal threshold."""

    def is_applicable(self, observations: pa.typing.DataFrame[FlatObserved]) -> bool:
        return self.historical_observations is not None and has_season_buckets(observations)

    def _frame(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
        return outbreak_and_alert(self.historical_observations, observations, forecasts)


@metric()
class SensitivityMetric(_OutbreakMetric):
    """Sensitivity (true positive rate) for outbreak detection.

    Measures the proportion of actual outbreaks that were correctly
    predicted (alerted) by the forecast.

    Not a valid standalone optimization objective: it is maximised by alerting
    every period, so ``optimization_direction`` is deliberately unset.
    """

    spec = MetricSpec(
        metric_id="sensitivity",
        metric_name="Sensitivity",
        aggregation_op=AggregationOp.MEAN,
        description="True positive rate for outbreak detection alerts",
    )

    def compute_detailed(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
        frame = self._frame(observations, forecasts)
        outbreaks = frame[frame["outbreak"] == 1.0]
        return _as_metric(outbreaks, outbreaks["alert"])


@metric()
class SpecificityMetric(_OutbreakMetric):
    """Specificity (true negative rate) for outbreak detection.

    Measures the proportion of non-outbreak periods that were correctly
    not alerted by the forecast.

    Not a valid standalone optimization objective: it is maximised by never
    alerting, so ``optimization_direction`` is deliberately unset.
    """

    spec = MetricSpec(
        metric_id="specificity",
        metric_name="Specificity",
        aggregation_op=AggregationOp.MEAN,
        description="True negative rate for outbreak detection alerts",
    )

    def compute_detailed(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
        frame = self._frame(observations, forecasts)
        quiet = frame[frame["outbreak"] == 0.0]
        return _as_metric(quiet, 1.0 - quiet["alert"])


@metric()
class OutbreakAccuracyMetric(_OutbreakMetric):
    """Accuracy for outbreak detection.

    Measures the proportion of all periods where the alert status
    correctly matches the outbreak status: (TP + TN) / (TP + TN + FP + FN).

    Not a valid standalone optimization objective: outbreaks are rare, so it is
    close to maximised by never alerting. ``optimization_direction`` is
    deliberately unset.
    """

    spec = MetricSpec(
        metric_id="outbreak_accuracy",
        metric_name="Outbreak Accuracy",
        aggregation_op=AggregationOp.MEAN,
        description="Proportion of correctly classified outbreak/non-outbreak periods",
    )

    def compute_detailed(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
        frame = self._frame(observations, forecasts)
        return _as_metric(frame, (frame["alert"] == frame["outbreak"]).astype(float))
