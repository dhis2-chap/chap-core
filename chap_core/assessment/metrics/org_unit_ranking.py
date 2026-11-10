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
    Spearman rank correlation between the forecast and observed incidence across org
    units, computed separately for each time period and horizon and then averaged.

    Incidence is the median forecast and the observed cases divided by the org unit's
    population for that period, so large org units do not rank high just by being
    large. The metric needs population and is not applicable without it.

    A ranking only exists across org units, so the metric has no per-location value
    and cannot be broken down by location.
    """

    spec = MetricSpec(
        metric_id="org_unit_ranking",
        metric_name="Org Unit Ranking",
        output_dimensions=(DataDimension.time_period, DataDimension.horizon_distance),
        aggregation_op=AggregationOp.MEAN,
        description=(
            "Spearman rank correlation between forecast and observed incidence (cases per population) "
            "across org units, averaged over time periods. 1 means org units are ranked perfectly, "
            "0 means no better than random. Needs population data."
        ),
        optimization_direction=OptimizationDirection.MAXIMIZE,
        comparison_op="difference",
    )

    def compute_detailed(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
        keys = ["time_period", "horizon_distance"]
        median_forecasts = forecasts.groupby(["location", *keys], as_index=False)["forecast"].median()
        merged = median_forecasts.merge(
            observations[["location", "time_period", "disease_cases", "population"]],
            on=["location", "time_period"],
            how="inner",
        )
        merged = merged[merged["population"] > 0].copy()
        merged["forecast_incidence"] = merged["forecast"] / merged["population"]
        merged["observed_incidence"] = merged["disease_cases"] / merged["population"]
        rows = [
            {
                "time_period": time_period,
                "horizon_distance": horizon,
                "metric": _spearman(group["forecast_incidence"], group["observed_incidence"]),
            }
            for (time_period, horizon), group in merged.groupby(keys)
        ]
        return pd.DataFrame(rows, columns=[*keys, "metric"])

    def is_applicable(self, observations: pd.DataFrame) -> bool:
        """A ranking needs population and at least two org units that have it."""
        if "population" not in observations.columns:
            return False
        return observations.loc[observations["population"] > 0, "location"].nunique() > 1


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rank correlation, NaN when either side has no variation (e.g. no cases anywhere in a period)."""
    if a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(a.corr(b, method="spearman"))
