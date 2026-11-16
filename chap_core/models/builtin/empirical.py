"""Helpers for baselines that forecast with the empirical distribution of observed values."""

import numpy as np

from chap_core.models.builtin.registry import BaselineNotApplicable
from chap_core.time_period import Month, PeriodRange, Week

N_SAMPLES = 101


def even_samples(values: np.ndarray) -> np.ndarray:
    """``N_SAMPLES`` sorted samples: the values, each repeated about equally often, around their median.

    The middle sample is the exact median of the values, and the ``N_SAMPLES - 1`` others spread
    the values evenly, half on each side: sample ``i`` of them is the sorted value at position
    ``floor((i + 0.5) * k / (N_SAMPLES - 1))`` of the ``k`` values, mirrored around the middle so
    the samples are symmetric. With fewer values than that each value is repeated, with more the
    values are thinned evenly. So every sample except the middle one is an observed value, and
    the middle one is too unless ``k`` is even.
    """
    values = np.sort(np.asarray(values, dtype=float))
    n_spread = N_SAMPLES - 1
    positions = np.arange(n_spread)
    lower = (2 * positions + 1) * len(values) // (2 * n_spread)
    spread = values[np.where(positions < n_spread / 2, lower, len(values) - 1 - lower[::-1])]
    half = n_spread // 2
    return np.concatenate([spread[:half], [np.median(values)], spread[half:]])


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
