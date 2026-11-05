"""C-SUM threshold strategy: WHO three-period smoothed baseline (prev / current / next).

Malaria programmes form a cumulative-sum (C-SUM) endemic channel by pooling case counts
from a three-period window — the previous, target, and next sub-period — across several
baseline years, then taking the mean (and optionally the population SD or the upper 95%
limit mean + 1.96 * SD). This is not a running sum within a single year.

Like :mod:`percentile`, the channel is static: one dataset defines one threshold per
``(location, season)``, and requested periods only select which rows come back.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from chap_core.assessment.thresholds import threshold
from chap_core.assessment.thresholds.base import ThresholdStrategyBase, align_seasonal_to_periods
from chap_core.assessment.thresholds.params import CSumLine, CSumParams
from chap_core.assessment.thresholds.percentile import filter_to_baseline
from chap_core.time_period.vectorized import extract_year, season_column

# Match CAPS MIN_CSUM_WINDOW_VALUES: need at least one full three-period window.
_MIN_WINDOW_VALUES = 3

_SEASON_COUNT = {"month": 12, "week": 52}


def _shift_season(year: int, season: int, delta: int, n_seasons: int) -> tuple[int, int]:
    """Shift a 1-based season index by ``delta``, wrapping across years."""
    idx = season - 1 + delta
    while idx < 0:
        year -= 1
        idx += n_seasons
    while idx >= n_seasons:
        year += 1
        idx -= n_seasons
    return year, idx + 1


def _population_std(values: list[float]) -> float:
    """Population standard deviation (divide by N). Empty → NaN; single value → 0."""
    if len(values) == 0:
        return float("nan")
    if len(values) == 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _line_value(kind: CSumLine, window_values: list[float]) -> float:
    """Map a C-SUM line kind onto the pooled window values."""
    if len(window_values) == 0:
        return float("nan")
    mean = sum(window_values) / len(window_values)
    if kind == "mean":
        return mean
    sd = _population_std(window_values)
    if kind == "sd":
        return sd
    return mean + 1.96 * sd


def collect_window_values(
    lookup: dict[tuple[str, int, int], float],
    location: str,
    year: int,
    season: int,
    n_seasons: int,
) -> list[float]:
    """Collect prev / current / next season values for one (location, year, season)."""
    values: list[float] = []
    for delta in (-1, 0, 1):
        y, s = _shift_season(year, season, delta, n_seasons)
        value = lookup.get((location, y, s))
        if value is not None:
            values.append(value)
    return values


def compute_c_sum_thresholds(
    historical_observations: pd.DataFrame,
    line_kinds: list[CSumLine],
    *,
    lookup_observations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute C-SUM threshold lines per (location, season).

    Parameters
    ----------
    historical_observations :
        Baseline observations used to decide which ``(location, year, season)``
        combinations contribute to the pool. Columns
        ``[location, time_period, disease_cases]``.
    line_kinds :
        Ordered list of statistics to emit (``mean``, ``sd``, ``mean_plus_1_96_sd``).
    lookup_observations :
        Frame used to resolve adjacent-period values. Defaults to
        ``historical_observations``. Pass a wider frame so January of the first
        baseline year can still pick up the prior December.

    Returns
    -------
    pd.DataFrame
        Columns ``[location, month|week, line, threshold]``, where ``line`` is the
        zero-based index into ``line_kinds``. Seasons with fewer than three pooled
        window values are omitted.
    """
    if historical_observations.empty:
        return pd.DataFrame(columns=["location", "month", "line", "threshold"])

    lookup_frame = lookup_observations if lookup_observations is not None else historical_observations
    season_name, _ = season_column(historical_observations["time_period"])
    n_seasons = _SEASON_COUNT[season_name]

    lookup_years = extract_year(lookup_frame["time_period"])
    _, lookup_buckets = season_column(lookup_frame["time_period"])
    lookup: dict[tuple[str, int, int], float] = {}
    for loc, year, bucket, cases in zip(
        lookup_frame["location"],
        lookup_years,
        lookup_buckets,
        lookup_frame["disease_cases"],
        strict=True,
    ):
        if pd.isna(cases):
            continue
        lookup[(str(loc), int(year), int(bucket))] = float(cases)

    baseline_years = extract_year(historical_observations["time_period"])
    _, baseline_buckets = season_column(historical_observations["time_period"])
    # Unique (location, season) pairs that appear in the baseline, plus the years
    # that contribute for each location.
    pairs = (
        pd.DataFrame(
            {
                "location": historical_observations["location"].astype(str),
                "year": np.asarray(baseline_years, dtype=int),
                season_name: np.asarray(baseline_buckets, dtype=int),
            }
        )
        .drop_duplicates()
        .groupby(["location", season_name], sort=False)["year"]
        .apply(list)
        .reset_index(name="years")
    )

    rows: list[dict] = []
    for _, record in pairs.iterrows():
        location = str(record["location"])
        season = int(record[season_name])
        years = [int(year) for year in record["years"]]
        window_values: list[float] = []
        for year in years:
            window_values.extend(collect_window_values(lookup, location, year, season, n_seasons))
        if len(window_values) < _MIN_WINDOW_VALUES:
            continue
        for line_index, kind in enumerate(line_kinds):
            rows.append(
                {
                    "location": location,
                    season_name: season,
                    "line": line_index,
                    "threshold": _line_value(kind, window_values),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["location", season_name, "line", "threshold"])
    return pd.DataFrame(rows)[["location", season_name, "line", "threshold"]]


@threshold(
    "c_sum",
    "C-SUM (WHO)",
    CSumParams,
    "WHO cumulative-sum endemic channel: mean (and optionally population SD or mean + 1.96 "
    "population SD) of a three-period window (previous, target, and next sub-period) pooled "
    "across the most recent complete baseline years. Defaults to the upper 95% limit over 5 years.",
)
class CSumThresholdStrategy(ThresholdStrategyBase[CSumParams]):
    """Registered strategy computing one C-SUM line per requested statistic."""

    def compute(
        self,
        historical_observations: pd.DataFrame,
        period_ids: list[str],
        params: CSumParams,
    ) -> pd.DataFrame:
        windowed = filter_to_baseline(historical_observations, params.baseline_years)
        per_season = compute_c_sum_thresholds(
            windowed,
            params.lines,
            lookup_observations=historical_observations,
        )
        return align_seasonal_to_periods(per_season, period_ids)
