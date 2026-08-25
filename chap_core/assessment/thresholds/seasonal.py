"""Seasonal threshold strategy: mean + k*std over historical same-season (month or week) values."""

from __future__ import annotations

import pandas as pd

from chap_core.assessment.thresholds import threshold
from chap_core.assessment.thresholds.base import ThresholdStrategyBase


def _extract_month(time_period: pd.Series) -> pd.Index:
    """Vectorised month-of-year extraction from monthly time_period strings.

    Equivalent to ``time_period.apply(lambda tp: TimePeriod.parse(str(tp)).start_timestamp.month)``
    but ~100x faster on large frames. Only valid for monthly periods.
    """
    return pd.PeriodIndex(time_period.astype(str), freq="M").month


def _extract_week(time_period: pd.Series) -> pd.Index:
    """Vectorised week-of-year extraction from weekly time_period strings.

    Accepts every weekly format ``TimePeriod.parse`` supports (``2016W01``, ``2016-W01``,
    ``2016SunW01``, ``2016-S01``), with both zero-padded and unpadded week numbers
    (``2016W01`` and ``2016W1`` map to the same week).
    """
    values = time_period.astype(str)
    weeks = values.str.extract(r"[WS](\d{1,2})$", expand=False)
    if weeks.isna().any():
        bad = values[weeks.isna()].iloc[0]
        raise ValueError(f"Cannot parse week from time period: {bad}")
    return pd.Index(weeks.astype(int))


def _season_column(time_period: pd.Series) -> tuple[str, pd.Index]:
    """Return the season bucket for a period series: ("week", week-of-year) for weekly periods, ("month", month-of-year) otherwise."""
    first = str(time_period.iloc[0])
    if "W" in first or "-S" in first:
        return "week", _extract_week(time_period)
    return "month", _extract_month(time_period)


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
    season, buckets = _season_column(df["time_period"])
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
        season, buckets = _season_column(requested["period_id"])
        if season not in per_season.columns:
            raise ValueError(f"period_ids are {season}-based but the dataset's time periods have a different frequency")
        requested[season] = buckets
        merged = requested.merge(per_season, on=season)
        return merged[["period_id", "location", "threshold"]]
