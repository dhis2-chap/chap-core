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
from chap_core.assessment.outbreak.threshold_model import ThresholdOutbreakModel
from chap_core.time_period.vectorized import season_column

if TYPE_CHECKING:
    import pandera.pandas as pa

    from chap_core.assessment.flat_representations import FlatObserved
    from chap_core.assessment.outbreak.base import OutbreakModelBase

#: Fraction of forecast samples that must exceed the threshold for an alert to be raised.
ALERT_SAMPLE_FRACTION = 0.5

_OUTBREAK_COLUMNS = ["location", "time_period", "horizon_distance", "outbreak", "alert"]
_METRIC_DIMENSIONS = ["location", "time_period", "horizon_distance"]


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
    model: OutbreakModelBase | None = None,
) -> pd.DataFrame:
    """Label every scored cell as outbreak and alert against the epidemic channel.

    The channel does both jobs: it labels an observation an outbreak, and the
    outbreak model scores forecasts against the same line. Alerting and labelling
    therefore cannot drift apart, which is what keeps Brier and log score proper.

    Args:
        historical_observations: Observations the channel is computed from, with
            columns ``[location, time_period, disease_cases]``. ``None`` yields an
            empty frame -- the metrics are not applicable without it.
        observations: Observed cases to label, same columns.
        forecasts: Forecast samples, with columns
            ``[location, time_period, horizon_distance, sample, forecast]``.
        model: Outbreak model producing the alert probabilities. Defaults to
            :class:`ThresholdOutbreakModel` on the seasonal mean + 2*std channel.

    Returns:
        One row per ``(location, time_period, horizon_distance)`` that has both a
        computable threshold and a forecast, with columns
        ``[location, time_period, horizon_distance, outbreak, alert]``. ``outbreak``
        is 1.0 where observed cases exceed the channel; ``alert`` is 1.0 where more
        than :data:`ALERT_SAMPLE_FRACTION` of the samples exceed it. Cells whose
        threshold cannot be computed (a single historical value gives an undefined
        standard deviation) are dropped.
    """
    empty = pd.DataFrame(columns=_OUTBREAK_COLUMNS)
    if historical_observations is None or historical_observations.empty:
        return empty
    if observations.empty or forecasts.empty:
        return empty

    model = model or ThresholdOutbreakModel()
    target_periods = sorted(set(forecasts["time_period"].astype(str)))
    try:
        probabilities = model.alert_probabilities(historical_observations, target_periods, forecasts=forecasts)
    except ValueError:
        # Historical and scored periods are at different frequencies; nothing to join on.
        return empty
    if probabilities.empty:
        return empty

    probabilities["alert"] = (probabilities["probability"] > ALERT_SAMPLE_FRACTION).astype(float)
    channels = probabilities[["location", "time_period", "threshold"]].drop_duplicates()

    labelled = observations[["location", "time_period", "disease_cases"]].merge(
        channels, on=["location", "time_period"], how="inner"
    )
    if labelled.empty:
        return empty
    labelled["outbreak"] = (labelled["disease_cases"] > labelled["threshold"]).astype(float)

    merged = labelled[["location", "time_period", "outbreak"]].merge(
        probabilities[[*_METRIC_DIMENSIONS, "alert"]],
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
