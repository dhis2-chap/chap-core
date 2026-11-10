"""
Org unit ranking metric: how well the forecasts order org units by disease burden.
"""

import pandas as pd

from chap_core.assessment.flat_representations import DataDimension
from chap_core.assessment.metrics import metric
from chap_core.assessment.metrics.base import (
    AggregationOp,
    Metric,
    MetricSpec,
    OptimizationDirection,
)


@metric()
class OrgUnitRankingMetric(Metric):
    """
    Spearman rank correlation between the median forecast and the observed cases
    across org units, computed separately for each time period and horizon and then
    averaged.

    A ranking only exists across org units, so the metric has no per-location value
    and cannot be broken down by location.
    """

    spec = MetricSpec(
        metric_id="org_unit_ranking",
        metric_name="Org Unit Ranking",
        output_dimensions=(DataDimension.time_period, DataDimension.horizon_distance),
        aggregation_op=AggregationOp.MEAN,
        description=(
            "Spearman rank correlation between forecast and observed cases across org units, "
            "averaged over time periods. 1 means org units are ranked perfectly, 0 means no better than random."
        ),
        optimization_direction=OptimizationDirection.MAXIMIZE,
        comparison_op="difference",
    )

    def compute_detailed(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
        keys = ["time_period", "horizon_distance"]
        median_forecasts = forecasts.groupby(["location", *keys], as_index=False)["forecast"].median()
        merged = median_forecasts.merge(
            observations[["location", "time_period", "disease_cases"]],
            on=["location", "time_period"],
            how="inner",
        )
        rows = [
            {
                "time_period": time_period,
                "horizon_distance": horizon,
                "metric": group["forecast"].corr(group["disease_cases"], method="spearman")
                if len(group) > 1
                else float("nan"),
            }
            for (time_period, horizon), group in merged.groupby(keys)
        ]
        return pd.DataFrame(rows, columns=[*keys, "metric"])

    def is_applicable(self, observations: pd.DataFrame) -> bool:
        """A ranking needs at least two org units."""
        return observations["location"].nunique() > 1
