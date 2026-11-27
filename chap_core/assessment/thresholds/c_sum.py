"""C-SUM threshold strategy: WHO three-period smoothed endemic channel (prev / current / next).

Malaria programmes form a cumulative-sum (C-SUM) endemic channel by pooling case counts
from a three-period window (the previous, target and next sub-period) across several
baseline years. The channel is the mean of the pooled values plus ``k`` times their
population standard deviation (divide by N, as the CAPS implementation does; the
``seasonal`` strategy uses the sample standard deviation). This is not a running sum
within a single year.

Like :mod:`percentile`, the channel is static: one dataset defines one threshold per
``(location, season)``, and requested periods only select which rows come back.

A window counts only when all three of its periods have an observation inside the
baseline years, so a missing period never shrinks a window, neighbouring years are
never pulled in from outside the baseline, and a partial final year never leaks in.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from chap_core.assessment.thresholds import threshold
from chap_core.assessment.thresholds.base import ThresholdStrategyBase, align_seasonal_to_periods
from chap_core.assessment.thresholds.params import CSumParams
from chap_core.assessment.thresholds.percentile import filter_to_baseline, last_complete_year
from chap_core.time_period.vectorized import extract_year, season_column


def _period_ordinals(years: pd.Series, buckets: pd.Series, season: str) -> pd.Series:
    """Number consecutive periods so that ``ordinal + 1`` is always the next period.

    Months count ``year * 12 + month``. Weeks follow the ISO calendar, whose years have
    52 or 53 weeks, so the week after ``2020-W53`` is ``2021-W01``.
    """
    if season == "month":
        return years * 12 + buckets
    pairs = pd.DataFrame({"year": years, "bucket": buckets}).drop_duplicates()
    pairs["ordinal"] = [
        date.fromisocalendar(int(year), int(week), 1).toordinal() // 7
        for year, week in zip(pairs["year"], pairs["bucket"], strict=True)
    ]
    keyed = pd.DataFrame({"year": years, "bucket": buckets}).merge(pairs, on=["year", "bucket"], how="left")
    return pd.Series(keyed["ordinal"].to_numpy(), index=years.index)


def compute_c_sum_thresholds(
    historical_observations: pd.DataFrame,
    std_multipliers: list[float],
    *,
    season_name: str = "month",
) -> pd.DataFrame:
    """Compute C-SUM threshold lines per (location, season).

    Args:
        historical_observations: Baseline observations with columns
            ``[location, time_period, disease_cases]``. Windows are formed only from
            these rows, so the caller decides which years contribute.
        std_multipliers: ``k`` values; each produces one line ``mean + k * population SD``.
        season_name: Season column to emit when the input is empty and the frequency
            cannot be read from it.

    Returns:
        DataFrame with columns ``[location, month|week, line, threshold]``, where ``line``
        is the index into ``std_multipliers``. Seasons without a complete three-period
        window are omitted.
    """
    observed = historical_observations.dropna(subset=["disease_cases"])
    if observed.empty:
        return pd.DataFrame(columns=["location", season_name, "line", "threshold"])

    season_name, bucket_index = season_column(observed["time_period"])
    years = pd.Series(extract_year(observed["time_period"]), index=observed.index, dtype=int)
    buckets = pd.Series(bucket_index, index=observed.index, dtype=int)
    observations = pd.DataFrame(
        {
            "location": observed["location"].astype(str),
            "ordinal": _period_ordinals(years, buckets, season_name),
            season_name: buckets,
            "cases": observed["disease_cases"].astype(float),
        }
    )

    def neighbour(offset: int, name: str) -> pd.DataFrame:
        shifted = observations[["location", "ordinal", "cases"]].rename(columns={"cases": name})
        shifted["ordinal"] = shifted["ordinal"] + offset
        return shifted

    windows = observations.merge(neighbour(1, "previous"), on=["location", "ordinal"]).merge(
        neighbour(-1, "next"), on=["location", "ordinal"]
    )
    if windows.empty:
        return pd.DataFrame(columns=["location", season_name, "line", "threshold"])

    pooled = pd.concat(
        [
            windows[["location", season_name, column]].rename(columns={column: "value"})
            for column in ("cases", "previous", "next")
        ]
    )
    stats = pooled.groupby(["location", season_name])["value"].agg(
        mean="mean",
        std=lambda values: values.std(ddof=0),
    )
    lines = []
    for index, multiplier in enumerate(std_multipliers):
        line = (stats["mean"] + multiplier * stats["std"]).rename("threshold").reset_index()
        line["line"] = index
        lines.append(line)
    result: pd.DataFrame = pd.concat(lines, ignore_index=True)[["location", season_name, "line", "threshold"]]
    return result


@threshold(
    "c_sum",
    "C-SUM (WHO)",
    CSumParams,
    "WHO cumulative-sum endemic channel: mean + k x population SD of a three-period window "
    "(previous, target and next sub-period) pooled across the most recent complete baseline "
    "years. Uses the population SD (divide by N, as CAPS does), unlike seasonal, which uses the "
    "sample SD. Defaults to k = 1.96 (upper 95% limit) over 5 years.",
)
class CSumThresholdStrategy(ThresholdStrategyBase[CSumParams]):
    """Registered strategy computing one C-SUM line per requested std multiplier."""

    def compute(
        self,
        historical_observations: pd.DataFrame,
        period_ids: list[str],
        params: CSumParams,
    ) -> pd.DataFrame:
        requested_season = season_column(pd.Series(period_ids))[0] if period_ids else "month"
        if historical_observations.empty:
            return align_seasonal_to_periods(
                compute_c_sum_thresholds(historical_observations, params.lines, season_name=requested_season),
                period_ids,
            )
        windowed = filter_to_baseline(historical_observations, params.baseline_years)
        if params.baseline_years is None:
            # Even with all history, a partial final year must not leak into the windows.
            last = last_complete_year(historical_observations)
            windowed = windowed[extract_year(windowed["time_period"]) <= last]
        return align_seasonal_to_periods(compute_c_sum_thresholds(windowed, params.lines), period_ids)
