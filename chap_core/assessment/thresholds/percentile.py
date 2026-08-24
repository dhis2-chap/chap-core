"""Percentile threshold strategy: WHO-style endemic channel from historical same-season values.

Malaria surveillance programmes plot channel graphs from a percentile of the previous few
complete years, reading a value above the upper line as an outbreak. A percentile is used
rather than mean + k*std because case counts are heavily right-skewed and a single past
epidemic year inflates the standard deviation far above any plausible endemic level.
"""

from __future__ import annotations

import pandas as pd

from chap_core.assessment.thresholds import threshold
from chap_core.assessment.thresholds.base import ThresholdStrategyBase
from chap_core.assessment.thresholds.period_buckets import extract_year, season_column

DEFAULT_QUANTILE = 0.75
DEFAULT_LOOKBACK_YEARS = 5


def filter_to_lookback(
    historical_observations: pd.DataFrame,
    period_ids: list[str],
    lookback_years: int | None = DEFAULT_LOOKBACK_YEARS,
) -> pd.DataFrame:
    """Restrict observations to the complete years preceding the periods being requested.

    The window is the ``lookback_years`` years before the latest requested period's year. That
    year is itself excluded: a channel for the current year is built from previous complete
    years, so an outbreak in progress cannot raise its own threshold. Pass ``None`` to use all
    available history.
    """
    if lookback_years is None:
        return historical_observations

    anchor = int(extract_year(pd.Series(period_ids)).max())
    first_year = anchor - int(lookback_years)
    years = extract_year(historical_observations["time_period"])
    windowed = historical_observations[(years >= first_year) & (years < anchor)]

    if windowed.empty:
        raise ValueError(
            f"No observations in the {lookback_years}-year window {first_year}-{anchor - 1} "
            f"for requested periods up to {anchor}; the data covers {int(years.min())}-{int(years.max())}"
        )
    return windowed


def compute_percentile_thresholds(
    historical_observations: pd.DataFrame, quantile: float = DEFAULT_QUANTILE
) -> pd.DataFrame:
    """Compute outbreak thresholds as a per-season percentile of historical observations.

    Args:
        historical_observations: DataFrame with columns [location, time_period, disease_cases]
        quantile: Percentile to take, as a fraction (default 0.75, the WHO channel's upper line).

    Returns:
        DataFrame with columns [location, month, threshold] for monthly data,
        or [location, week, threshold] for weekly data.
    """
    df = historical_observations.copy()
    season, buckets = season_column(df["time_period"])
    df[season] = buckets
    grouped = df.groupby(["location", season])["disease_cases"].quantile(quantile).reset_index()
    grouped = grouped.rename(columns={"disease_cases": "threshold"})
    return grouped[["location", season, "threshold"]]


@threshold(
    "percentile",
    "Seasonal percentile (WHO endemic channel)",
    "Outbreak threshold as a percentile of historical same-month (or same-week) values, over a "
    "trailing window of complete years. Defaults to the 75th percentile over 5 years, following "
    "WHO malaria channel practice. Robust to past epidemic years, which inflate mean + k*std.",
)
class PercentileThresholdStrategy(ThresholdStrategyBase):
    """Wraps :func:`compute_percentile_thresholds` as a registered strategy."""

    def compute(
        self,
        historical_observations: pd.DataFrame,
        period_ids: list[str],
        params: dict | None = None,
    ) -> pd.DataFrame:
        params = params or {}
        quantile = float(params.get("quantile", DEFAULT_QUANTILE))
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"quantile must be between 0 and 1, got {quantile}")
        lookback_years = params.get("lookback_years", DEFAULT_LOOKBACK_YEARS)

        windowed = filter_to_lookback(historical_observations, period_ids, lookback_years)
        per_season = compute_percentile_thresholds(windowed, quantile=quantile)
        requested = pd.DataFrame({"period_id": period_ids})
        season, buckets = season_column(requested["period_id"])
        if season not in per_season.columns:
            raise ValueError(f"period_ids are {season}-based but the dataset's time periods have a different frequency")
        requested[season] = buckets
        merged = requested.merge(per_season, on=season)
        return merged[["period_id", "location", "threshold"]]
