"""Seasonal threshold strategy: mean + k*std over historical same-month values."""

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


def compute_seasonal_thresholds(historical_observations: pd.DataFrame, k: float = 2.0) -> pd.DataFrame:
    """Compute outbreak thresholds from historical observations.

    Args:
        historical_observations: DataFrame with columns [location, time_period, disease_cases]
        k: Number of standard deviations above the mean (default 2.0).

    Returns:
        DataFrame with columns [location, month, threshold]
    """
    df = historical_observations.copy()
    df["month"] = _extract_month(df["time_period"])
    grouped = df.groupby(["location", "month"])["disease_cases"].agg(["mean", "std"]).reset_index()
    grouped["threshold"] = grouped["mean"] + k * grouped["std"]
    return grouped[["location", "month", "threshold"]]


@threshold(
    "seasonal",
    "Seasonal mean + k*std",
    "Outbreak threshold as mean + k standard deviations of historical same-month values.",
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
        per_month = compute_seasonal_thresholds(historical_observations, k=k)
        requested = pd.DataFrame({"period_id": period_ids})
        requested["month"] = _extract_month(requested["period_id"])
        merged = requested.merge(per_month, on="month")
        merged = merged.rename(columns={"location": "org_unit"})
        return merged[["period_id", "org_unit", "threshold"]]
