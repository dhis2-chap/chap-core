"""Percentile threshold strategy: WHO-style endemic channel from historical same-season values.

Malaria surveillance programmes plot channel graphs from a percentile of the previous few
complete years, reading a value above the upper line as an outbreak. A percentile is used
rather than mean + k*std because case counts are heavily right-skewed and a single past
epidemic year inflates the standard deviation far above any plausible endemic level.
"""

from __future__ import annotations

import pandas as pd

from chap_core.assessment.thresholds import threshold
from chap_core.assessment.thresholds.base import ThresholdStrategyBase, align_seasonal_to_periods
from chap_core.assessment.thresholds.params import PercentileParams, line_values
from chap_core.time_period.vectorized import extract_year, season_column


def filter_to_baseline(
    historical_observations: pd.DataFrame,
    period_ids: list[str],
    baseline_years: int | None,
) -> pd.DataFrame:
    """Restrict observations to the complete years preceding the periods being requested.

    The baseline window is anchored on the latest requested period's year: it spans the
    ``baseline_years`` years before that year, and all requested periods share this one
    window. The anchor year is itself excluded, so an outbreak in progress cannot raise
    its own threshold. When a request spans a year boundary, the earlier periods' own
    year is still inside the window. Pass ``None`` to use all available history.
    """
    if baseline_years is None:
        return historical_observations

    anchor = int(extract_year(pd.Series(period_ids)).max())
    first_year = anchor - baseline_years
    years = extract_year(historical_observations["time_period"])
    windowed = historical_observations[(years >= first_year) & (years < anchor)]

    if windowed.empty:
        raise ValueError(
            f"No observations in the {baseline_years}-year baseline window {first_year}-{anchor - 1} "
            f"for requested periods up to {anchor}; the data covers {int(years.min())}-{int(years.max())}"
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
    "baseline window of complete years. Defaults to the 75th percentile over 5 years, the WHO "
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
        windowed = filter_to_baseline(historical_observations, period_ids, params.baseline_years)
        per_season = compute_percentile_thresholds(windowed, line_values(params.quantile))
        return align_seasonal_to_periods(per_season, period_ids)
