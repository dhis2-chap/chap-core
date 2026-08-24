"""Seasonal threshold strategy: mean + k*std over historical same-season (month or week) values."""

from __future__ import annotations

import pandas as pd

from chap_core.assessment.thresholds import threshold
from chap_core.assessment.thresholds.base import ThresholdStrategyBase
from chap_core.assessment.thresholds.period_buckets import season_column


def compute_seasonal_thresholds(historical_observations: pd.DataFrame, k: float = 2.0) -> pd.DataFrame:
    """Compute outbreak thresholds from historical observations.

    Args:
        historical_observations: DataFrame with columns [location, time_period, disease_cases]
        k: Number of standard deviations above the mean (default 2.0).

    Returns:
        DataFrame with columns [location, month, threshold] for monthly data,
        or [location, week, threshold] for weekly data.
    """
    df = historical_observations.copy()
    season, buckets = season_column(df["time_period"])
    df[season] = buckets
    grouped = df.groupby(["location", season])["disease_cases"].agg(["mean", "std"]).reset_index()
    grouped["threshold"] = grouped["mean"] + k * grouped["std"]
    return grouped[["location", season, "threshold"]]


@threshold(
    "seasonal",
    "Seasonal mean + k*std",
    "Outbreak threshold as mean + k standard deviations of historical same-month (or same-week) values.",
)
class SeasonalThresholdStrategy(ThresholdStrategyBase):
    """Wraps :func:`compute_seasonal_thresholds` as a registered strategy."""

    def compute(
        self,
        historical_observations: pd.DataFrame,
        period_ids: list[str],
        params: dict | None = None,
    ) -> pd.DataFrame:
        k = float((params or {}).get("k", 2.0))
        per_season = compute_seasonal_thresholds(historical_observations, k=k)
        requested = pd.DataFrame({"period_id": period_ids})
        season, buckets = season_column(requested["period_id"])
        if season not in per_season.columns:
            raise ValueError(f"period_ids are {season}-based but the dataset's time periods have a different frequency")
        requested[season] = buckets
        merged = requested.merge(per_season, on=season)
        return merged[["period_id", "location", "threshold"]]
