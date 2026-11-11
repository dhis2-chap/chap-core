"""Helpers for baselines that forecast with the empirical distribution of observed values."""

import numpy as np

from chap_core.models.builtin.registry import BaselineNotApplicable
from chap_core.time_period import Month, PeriodRange, Week


def even_samples(values: np.ndarray, n_samples: int) -> np.ndarray:
    """``n_samples`` samples that are the sorted values, each repeated about equally often.

    Sample ``i`` is the sorted value at position ``floor((i + 0.5) * k / n_samples)`` of the
    ``k`` values, mirrored around the middle so the samples are symmetric. With fewer values
    than samples each value is repeated, with more the values are thinned evenly. So the
    samples are always observed values, and with exactly ``n_samples`` values they are the
    values themselves. With at least as many samples as values, the median of the samples is
    the median of the values, except when ``n_samples`` is odd and ``k`` even, where it is the
    larger of the two middle values.
    """
    values = np.sort(np.asarray(values, dtype=float))
    positions = np.arange(n_samples)
    lower = (2 * positions + 1) * len(values) // (2 * n_samples)
    index = np.where(positions < n_samples / 2, lower, len(values) - 1 - lower[::-1])
    return values[index]


def periods_per_year(time_period: PeriodRange) -> int:
    """12 for monthly and 52 for weekly periods, the number of distinct values of ``period_of_year``."""
    if isinstance(time_period[0], Month):
        return 12
    if isinstance(time_period[0], Week):
        return 52
    raise BaselineNotApplicable(f"Seasonal baselines need monthly or weekly data, got {type(time_period[0]).__name__}")


def period_of_year(time_period: PeriodRange) -> np.ndarray:
    """The month (1-12) or week (1-52) of each period. Week 53 counts as week 52."""
    if periods_per_year(time_period) == 12:
        return np.asarray(time_period.month)
    return np.asarray(np.minimum(time_period.week, 52))
