"""Proper scoring rules for outbreak alert probabilities.

These score the exceedance probability itself rather than the alert decision, so
a model that says 0.6 is treated differently from one that says 0.95. Because the
channel that labels an outbreak is the same one the forecast is scored against,
the probability is a probability of exactly the labelled event -- which is what
makes these scores proper rather than merely numeric.

Brier skill is the number worth leading with. Outbreaks are rare, so a raw score
looks excellent for a model that never alerts; skill prices it against the
climatological base rate instead of against nothing.
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

#: Probabilities are clipped this far from 0 and 1 before taking a logarithm, so a
#: confident miss costs a large finite amount rather than infinity.
LOG_SCORE_CLIP = 1e-6


@metric()
class BrierScoreMetric(OutbreakScoredMixin, Metric):
    """Brier score: the mean squared error of the alert probability.

    Zero is perfect, one is a confidently wrong forecast every time. Decomposes
    per cell, so it can be reported per location, period or horizon.
    """

    spec = MetricSpec(
        metric_id="outbreak_brier",
        metric_name="Outbreak Brier Score",
        aggregation_op=AggregationOp.MEAN,
        description="Mean squared error of the outbreak alert probability",
        optimization_direction=OptimizationDirection.MINIMIZE,
    )

    def compute_detailed(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
        frame = self.outbreak_frame(observations, forecasts)
        return _as_metric(frame, (frame["probability"] - frame["outbreak"]) ** 2)


@metric()
class OutbreakLogScoreMetric(OutbreakScoredMixin, Metric):
    """Logarithmic score of the alert probability, clipped at :data:`LOG_SCORE_CLIP`.

    Punishes confident misses far harder than Brier does, which is the point: it
    is the score that notices a model asserting certainty it has not earned. The
    clip is what keeps a single confident miss from being infinite, so read this
    alongside Brier rather than instead of it.
    """

    spec = MetricSpec(
        metric_id="outbreak_log_score",
        metric_name="Outbreak Log Score",
        aggregation_op=AggregationOp.MEAN,
        description="Negative log likelihood of the observed outbreak status under the alert probability",
        optimization_direction=OptimizationDirection.MINIMIZE,
    )

    def compute_detailed(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
        frame = self.outbreak_frame(observations, forecasts)
        probability = frame["probability"].clip(LOG_SCORE_CLIP, 1 - LOG_SCORE_CLIP)
        outbreak = frame["outbreak"]
        score = -(outbreak * np.log(probability) + (1 - outbreak) * np.log(1 - probability))
        return _as_metric(frame, score)


@metric()
class BrierSkillScoreMetric(OutbreakScoredMixin, GlobalOnlyMetric):
    """Brier skill against climatology: how much the forecast beats the base rate.

    One is perfect, zero means the forecast is no better than always predicting
    the historical outbreak frequency, and negative means it is worse than that.
    A ratio of aggregates, so there is no per-cell value.

    Undefined when every scored cell is an outbreak or none is: the climatological
    reference is then perfect and there is no skill to measure against it.
    """

    spec = MetricSpec(
        metric_id="outbreak_brier_skill",
        metric_name="Outbreak Brier Skill Score",
        output_dimensions=(),
        aggregation_op=AggregationOp.MEAN,
        description="Brier score improvement over a climatological base-rate forecast",
        optimization_direction=OptimizationDirection.MAXIMIZE,
    )

    def compute_global(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> float:
        frame = self.outbreak_frame(observations, forecasts)
        if frame.empty:
            return float("nan")
        outbreak = frame["outbreak"]
        base_rate = float(outbreak.mean())
        reference = base_rate * (1 - base_rate)
        if reference == 0:
            return float("nan")
        brier = float(((frame["probability"] - outbreak) ** 2).mean())
        return 1 - brier / reference
