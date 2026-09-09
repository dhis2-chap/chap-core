"""Binary classification metrics for outbreak alerts.

These score the alert decision, after the exceedance probability has been cut at
:data:`~chap_core.assessment.metrics.outbreak_detection.ALERT_SAMPLE_FRACTION`.

Only F1 and Matthews correlation carry an ``optimization_direction``. Precision
and false-alarm rate, like sensitivity and specificity before them, are each
maximised by a degenerate model -- precision by alerting once and only when
certain, false-alarm rate by never alerting at all -- so neither is valid as a
standalone objective even though both are informative alongside the others.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from chap_core.assessment.metrics import metric
from chap_core.assessment.metrics.base import (
    AggregationOp,
    GlobalOnlyMetric,
    Metric,
    MetricSpec,
    OptimizationDirection,
)
from chap_core.assessment.metrics.outbreak_detection import OutbreakScoredMixin, _as_metric

if TYPE_CHECKING:
    import pandas as pd


def _confusion(frame: pd.DataFrame) -> tuple[int, int, int, int]:
    """Counts of (true positive, false positive, true negative, false negative)."""
    alert = frame["alert"] == 1.0
    outbreak = frame["outbreak"] == 1.0
    return (
        int((alert & outbreak).sum()),
        int((alert & ~outbreak).sum()),
        int((~alert & ~outbreak).sum()),
        int((~alert & outbreak).sum()),
    )


@metric()
class OutbreakPrecisionMetric(OutbreakScoredMixin, Metric):
    """Precision: the share of raised alerts that turned out to be real outbreaks.

    The counterpart to sensitivity. A pipeline that alerts constantly scores well
    on sensitivity and badly here; one that alerts almost never does the reverse.
    """

    spec = MetricSpec(
        metric_id="outbreak_precision",
        metric_name="Outbreak Precision",
        aggregation_op=AggregationOp.MEAN,
        description="Share of raised alerts that were real outbreaks",
    )

    def compute_detailed(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
        frame = self.outbreak_frame(observations, forecasts)
        alerted = frame[frame["alert"] == 1.0]
        return _as_metric(alerted, alerted["outbreak"])


@metric()
class FalseAlarmRateMetric(OutbreakScoredMixin, Metric):
    """False-alarm rate: the share of quiet periods that were alerted anyway.

    One minus specificity, reported directly because it is the number an
    operational team feels -- how often a warning turns out to be nothing.
    """

    spec = MetricSpec(
        metric_id="false_alarm_rate",
        metric_name="False Alarm Rate",
        aggregation_op=AggregationOp.MEAN,
        description="Share of non-outbreak periods that raised an alert",
    )

    def compute_detailed(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
        frame = self.outbreak_frame(observations, forecasts)
        quiet = frame[frame["outbreak"] == 0.0]
        return _as_metric(quiet, quiet["alert"])


@metric()
class OutbreakF1Metric(OutbreakScoredMixin, GlobalOnlyMetric):
    """F1: the harmonic mean of precision and sensitivity.

    A ratio of aggregates, so it has no per-cell value. Unlike its two parts it
    prices both error types, which makes it usable as an objective.
    """

    spec = MetricSpec(
        metric_id="outbreak_f1",
        metric_name="Outbreak F1",
        output_dimensions=(),
        aggregation_op=AggregationOp.MEAN,
        description="Harmonic mean of outbreak precision and sensitivity",
        optimization_direction=OptimizationDirection.MAXIMIZE,
    )

    def compute_global(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> float:
        true_positive, false_positive, _, false_negative = _confusion(self.outbreak_frame(observations, forecasts))
        denominator = 2 * true_positive + false_positive + false_negative
        if denominator == 0:
            return float("nan")
        return 2 * true_positive / denominator


@metric()
class MatthewsCorrelationMetric(OutbreakScoredMixin, GlobalOnlyMetric):
    """Matthews correlation between alerts and outbreaks, in [-1, 1].

    Uses all four cells of the confusion matrix, so unlike F1 it does not ignore
    true negatives -- worth having when outbreaks are rare and quiet periods
    dominate. Zero means no better than chance.
    """

    spec = MetricSpec(
        metric_id="outbreak_mcc",
        metric_name="Outbreak Matthews Correlation",
        output_dimensions=(),
        aggregation_op=AggregationOp.MEAN,
        description="Correlation between alert and outbreak status across the full confusion matrix",
        optimization_direction=OptimizationDirection.MAXIMIZE,
    )

    def compute_global(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> float:
        true_positive, false_positive, true_negative, false_negative = _confusion(
            self.outbreak_frame(observations, forecasts)
        )
        denominator = np.sqrt(
            float(true_positive + false_positive)
            * float(true_positive + false_negative)
            * float(true_negative + false_positive)
            * float(true_negative + false_negative)
        )
        if denominator == 0:
            # A row or column of the matrix is empty, so correlation is undefined.
            return float("nan")
        return float((true_positive * true_negative - false_positive * false_negative) / denominator)
