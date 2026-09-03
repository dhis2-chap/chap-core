"""Vectorised season and year extraction from ``time_period`` strings.

Shared by the threshold strategies and the outbreak-detection metrics.
"""

from __future__ import annotations

import pandas as pd


def extract_month(time_period: pd.Series) -> pd.Index:
    """Vectorised month-of-year extraction from monthly time_period strings.

    Equivalent to ``time_period.apply(lambda tp: TimePeriod.parse(str(tp)).start_timestamp.month)``
    but ~100x faster on large frames. Only valid for monthly periods.
    """
    return pd.PeriodIndex(time_period.astype(str), freq="M").month


def extract_week(time_period: pd.Series) -> pd.Index:
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


def extract_year(time_period: pd.Series) -> pd.Index:
    """Vectorised year extraction from monthly or weekly time_period strings.

    Every supported format leads with a four-digit year (``2016-01``, ``2016W01``,
    ``2016-W01``, ``2016SunW01``, ``2016-S01``), so one pattern covers them all.
    """
    values = time_period.astype(str)
    years = values.str.extract(r"^(\d{4})", expand=False)
    if years.isna().any():
        bad = values[years.isna()].iloc[0]
        raise ValueError(f"Cannot parse year from time period: {bad}")
    return pd.Index(years.astype(int))


def season_column(time_period: pd.Series) -> tuple[str, pd.Index]:
    """Return the season bucket for a period series: ("week", week-of-year) for weekly periods, ("month", month-of-year) otherwise."""
    first = str(time_period.iloc[0])
    if "W" in first or "-S" in first:
        return "week", extract_week(time_period)
    return "month", extract_month(time_period)
