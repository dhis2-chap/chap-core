"""Helpers for baselines that forecast with the empirical distribution of observed values."""

import numpy as np

from chap_core.time_period import Month, PeriodRange, Week

N_SAMPLES = 100
# 99 quantile levels symmetric around 0.5, plus 0.5 once more. The two middle samples
# are then both the median, so the median of the samples is exactly the median of
# the values whichever median definition a metric uses.
QUANTILE_LEVELS = np.sort(np.append((np.arange(N_SAMPLES - 1) + 0.5) / (N_SAMPLES - 1), 0.5))


def quantile_samples(values: np.ndarray) -> np.ndarray:
    """``N_SAMPLES`` deterministic samples spread over the values, with their median in the middle."""
    return np.asarray(np.quantile(values, QUANTILE_LEVELS))


def period_of_year(time_period: PeriodRange) -> np.ndarray:
    """The month (1-12) or week (1-52) of each period. Week 53 counts as week 52."""
    if isinstance(time_period[0], Month):
        return np.asarray(time_period.month)
    if isinstance(time_period[0], Week):
        return np.asarray(np.minimum(time_period.week, 52))
    raise ValueError(f"Seasonal baselines need monthly or weekly data, got {type(time_period[0]).__name__}")
