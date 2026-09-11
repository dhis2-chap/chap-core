"""Geometric threshold strategy: the endemic channel on a log1p scale.

Case counts are strongly right-skewed, so an arithmetic mean is pulled up by single
epidemic years. This channel takes the mean and standard deviation of ``log1p(cases)``
instead and back-transforms with ``expm1``, giving a multiplicative channel: each line is
the geometric mean times the geometric standard deviation raised to the requested power.
Working on a ``x + 1`` scale keeps zero counts, which a plain geometric mean cannot
represent.

The centre of the channel is the geometric mean, which AM-GM puts at or below the
arithmetic mean for any non-constant history, so an epidemic year moves it far less. The
*width* is a different matter and is not uniformly tighter than mean + k*std: the log-scale
standard deviation is a relative spread, so on a short baseline a single extreme year can
make the back-transformed band wider than the arithmetic one. With a longer baseline
(roughly eight years or more for one 20x spike) the geometric band is the tighter of the
two.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from chap_core.assessment.thresholds import threshold
from chap_core.assessment.thresholds.base import ThresholdStrategyBase, align_seasonal_to_periods
from chap_core.assessment.thresholds.params import GeometricParams
from chap_core.time_period.vectorized import season_column


def log_seasonal_stats(historical_observations: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Per-(location, season) mean and std of ``log1p(disease_cases)``.

    Returns:
        The season column name ("month" or "week") and a DataFrame with columns
        ``[location, month|week, mean, std]``, both on the log1p scale.
    """
    df = historical_observations.assign(disease_cases=np.log1p(historical_observations["disease_cases"]))
    season, buckets = season_column(df["time_period"])
    df[season] = buckets
    grouped = df.groupby(["location", season])["disease_cases"].agg(["mean", "std"]).reset_index()
    return season, grouped


def compute_geometric_thresholds(historical_observations: pd.DataFrame, k: float = 2.0) -> pd.DataFrame:
    """Compute geometric (log1p) outbreak thresholds from historical observations.

    Args:
        historical_observations: DataFrame with columns [location, time_period, disease_cases]
        k: Number of geometric standard deviations above the geometric mean (default 2.0).

    Returns:
        DataFrame with columns [location, month, threshold] for monthly data,
        or [location, week, threshold] for weekly data.
    """
    season, stats = log_seasonal_stats(historical_observations)
    stats["threshold"] = np.expm1(stats["mean"] + k * stats["std"])
    return stats[["location", season, "threshold"]]


@threshold(
    "geometric",
    "Seasonal geometric mean + k*geometric SD",
    GeometricParams,
    "Outbreak threshold as expm1(mean(log1p(cases)) + k*std(log1p(cases))) over historical same-month "
    "(or same-week) values: the geometric mean times the geometric standard deviation to the power k. "
    "Centred on the geometric mean, which a past epidemic year moves far less than the arithmetic mean, "
    "and defined for zero counts. The band is multiplicative, so on a short baseline it can be wider than "
    "mean + k*std.",
)
class GeometricThresholdStrategy(ThresholdStrategyBase[GeometricParams]):
    """Registered strategy computing one line per requested std multiplier from shared log-scale stats."""

    def compute(
        self,
        historical_observations: pd.DataFrame,
        period_ids: list[str],
        params: GeometricParams,
    ) -> pd.DataFrame:
        season, stats = log_seasonal_stats(historical_observations)
        lines = []
        for i, multiplier in enumerate(params.lines):
            line = stats[["location", season]].copy()
            line["line"] = i
            line["threshold"] = np.expm1(stats["mean"] + multiplier * stats["std"])
            lines.append(line)
        per_season = pd.concat(lines, ignore_index=True)
        return align_seasonal_to_periods(per_season, period_ids)
