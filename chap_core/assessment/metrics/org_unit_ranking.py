"""
Org unit ranking metric: how well the forecasts order org units by disease burden.
"""

import numpy as np
import pandas as pd

from chap_core.assessment.flat_representations import DataDimension
from chap_core.assessment.forecast_cells import ForecastCells
from chap_core.assessment.metrics import metric
from chap_core.assessment.metrics.base import (
    AggregationOp,
    Metric,
    MetricSpec,
    OptimizationDirection,
)
from chap_core.i18n import N_


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
        description=N_(
            "Spearman rank correlation between forecast and observed incidence (cases per population) "
            "across org units, averaged over time periods. 1 means org units are ranked perfectly, "
            "0 means no better than random. Needs population data."
        ),
        optimization_direction=OptimizationDirection.MAXIMIZE,
        comparison_op="difference",
    )

    def compute_detailed_from_cells(self, cells: ForecastCells) -> pd.DataFrame:
        keys = ["time_period", "horizon_distance"]
        scored = cells.observed_only()
        if scored.population is None or not len(scored):
            return pd.DataFrame(columns=[*keys, "metric"])
        merged = scored.keys[keys].copy()
        merged["forecast"] = np.median(scored.samples, axis=1)
        merged["disease_cases"] = scored.observed
        merged["population"] = scored.population
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

    def compute_cell_metrics(self, samples: np.ndarray, observed: np.ndarray) -> np.ndarray:
        """A ranking compares org units, so there is no value per cell."""
        raise NotImplementedError("OrgUnitRankingMetric is computed across org units, not per cell")

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
