"""Seasonal threshold strategy: mean + k*std over historical same-season (month or week) values."""

from __future__ import annotations

import pandas as pd

from chap_core.assessment.thresholds import threshold
from chap_core.assessment.thresholds.base import ThresholdStrategyBase, align_seasonal_to_periods
from chap_core.assessment.thresholds.params import SeasonalParams, line_values
from chap_core.time_period.vectorized import season_column


def seasonal_stats(historical_observations: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Per-(location, season) mean and std of disease_cases.

    Returns:
        The season column name ("month" or "week") and a DataFrame with columns
        ``[location, month|week, mean, std]``.
    """
    df = historical_observations.copy()
    season, buckets = season_column(df["time_period"])
    df[season] = buckets
    grouped = df.groupby(["location", season])["disease_cases"].agg(["mean", "std"]).reset_index()
    return season, grouped


def compute_seasonal_thresholds(historical_observations: pd.DataFrame, k: float = 2.0) -> pd.DataFrame:
    """Compute outbreak thresholds from historical observations.

    Args:
        historical_observations: DataFrame with columns [location, time_period, disease_cases]
        k: Number of standard deviations above the mean (default 2.0).

    Returns:
        DataFrame with columns [location, month, threshold] for monthly data,
        or [location, week, threshold] for weekly data.
    """
    season, grouped = seasonal_stats(historical_observations)
    grouped["threshold"] = grouped["mean"] + k * grouped["std"]
    return grouped[["location", season, "threshold"]]


@threshold(
    "seasonal",
    "Seasonal mean + k*std",
    SeasonalParams,
    "Outbreak threshold as mean + k standard deviations of historical same-month (or same-week) values.",
)
class SeasonalThresholdStrategy(ThresholdStrategyBase[SeasonalParams]):
    """Registered strategy computing one line per requested std multiplier from shared seasonal stats."""

    def compute(
        self,
        historical_observations: pd.DataFrame,
        period_ids: list[str],
        params: SeasonalParams,
    ) -> pd.DataFrame:
        season, stats = seasonal_stats(historical_observations)
        lines = []
        for i, multiplier in enumerate(line_values(params.std_multiplier)):
            line = stats[["location", season]].copy()
            line["line"] = i
            line["threshold"] = stats["mean"] + multiplier * stats["std"]
            lines.append(line)
        per_season = pd.concat(lines, ignore_index=True)
        return align_seasonal_to_periods(per_season, period_ids)
