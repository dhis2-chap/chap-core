"""Percentile threshold strategy: WHO-style endemic channel from historical same-season values.

Malaria surveillance programmes plot channel graphs from a percentile of the previous few
complete years, reading a value above the upper line as an outbreak. A percentile is used
rather than mean + k*std because case counts are heavily right-skewed and a single past
epidemic year inflates the standard deviation far above any plausible endemic level.

The channel is static: one dataset defines one threshold per (location, season), computed
from the most recent complete years in the dataset, and the requested periods only select
which rows come back. Past and future periods therefore share one line. A consequence is
that a past period is scored against a baseline that includes it, so a historical epidemic
partly raises the line it is plotted against. That is inherent to the endemic-channel idiom.
"""

from __future__ import annotations

import pandas as pd

from chap_core.assessment.thresholds import threshold
from chap_core.assessment.thresholds.base import ThresholdStrategyBase, align_seasonal_to_periods
from chap_core.assessment.thresholds.params import PercentileParams, line_values
from chap_core.time_period.vectorized import extract_year, season_column

_FULL_YEAR_BUCKETS = {"month": 12, "week": 52}


def last_complete_year(historical_observations: pd.DataFrame) -> int:
    """Return the latest year in the data, or the year before it when the latest year is partial.

    A year is complete when it has every season bucket (12 months or at least 52 weeks).
    A live dataset's final year is usually in progress; including it would give the
    early-season buckets one more observation than the late-season ones, and would let an
    outbreak in progress raise the threshold it is about to be compared against.
    """
    years = extract_year(historical_observations["time_period"])
    season, buckets = season_column(historical_observations["time_period"])
    latest = int(years.max())
    latest_buckets = pd.Index(buckets)[years == latest].nunique()
    if latest_buckets < _FULL_YEAR_BUCKETS[season]:
        return latest - 1
    return latest


def filter_to_baseline(historical_observations: pd.DataFrame, baseline_years: int | None) -> pd.DataFrame:
    """Restrict observations to the most recent ``baseline_years`` complete years in the dataset.

    The window is anchored on the dataset, not on the requested periods, so every requested
    period, past or future, is compared against the same line. A partial final year is
    excluded (see :func:`last_complete_year`). Pass ``None`` to use all available history.
    """
    if baseline_years is None:
        return historical_observations

    years = extract_year(historical_observations["time_period"])
    last = last_complete_year(historical_observations)
    first = last - baseline_years + 1
    windowed = historical_observations[(years >= first) & (years <= last)]

    if windowed.empty:
        raise ValueError(
            f"No complete years in the {baseline_years}-year baseline window {first}-{last}; "
            f"the data covers {int(years.min())}-{int(years.max())}"
        )
    return windowed


def compute_percentile_thresholds(historical_observations: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    """Compute outbreak threshold lines as per-season percentiles of historical observations.

    All quantiles are computed from one groupby pass over the same observations.

    Args:
        historical_observations: DataFrame with columns [location, time_period, disease_cases]
        quantiles: Percentiles to take, as fractions (e.g. ``[0.25, 0.75]``).

    Returns:
        DataFrame with columns [location, month|week, line, threshold], where ``line``
        is the index into ``quantiles``.
    """
    df = historical_observations.copy()
    season, buckets = season_column(df["time_period"])
    df[season] = buckets
    unique_quantiles = list(dict.fromkeys(quantiles))
    per_quantile = df.groupby(["location", season])["disease_cases"].quantile(pd.Series(unique_quantiles)).unstack()
    lines = []
    for i, quantile in enumerate(quantiles):
        line = per_quantile[quantile].rename("threshold").reset_index()
        line["line"] = i
        lines.append(line)
    return pd.concat(lines, ignore_index=True)[["location", season, "line", "threshold"]]


@threshold(
    "percentile",
    "Seasonal percentile (WHO endemic channel)",
    PercentileParams,
    "Outbreak threshold as a percentile of historical same-month (or same-week) values, over a "
    "baseline window of the most recent complete years in the dataset. Defaults to the 75th "
    "percentile over 5 years, the WHO "
    "malaria channel practice. Robust to past epidemic years, which inflate mean + k*std.",
)
class PercentileThresholdStrategy(ThresholdStrategyBase[PercentileParams]):
    """Registered strategy computing one line per requested quantile from a shared baseline window."""

    def compute(
        self,
        historical_observations: pd.DataFrame,
        period_ids: list[str],
        params: PercentileParams,
    ) -> pd.DataFrame:
        windowed = filter_to_baseline(historical_observations, params.baseline_years)
        per_season = compute_percentile_thresholds(windowed, line_values(params.quantile))
        return align_seasonal_to_periods(per_season, period_ids)
