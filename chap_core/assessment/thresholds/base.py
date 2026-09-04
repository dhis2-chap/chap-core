"""Base class for threshold (endemic channel) calculation strategies.

A strategy turns a dataset's historical ``disease_cases`` observations into one
or more threshold lines per requested ``(period_id, org_unit)``. Subclasses
implement :meth:`compute`; registration happens via the :func:`threshold`
decorator, which also binds the strategy's typed params model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel


class ThresholdStrategyBase[ParamsT: BaseModel](ABC):
    """Base class for threshold strategies.

    Subclasses implement :meth:`compute`, which receives a flat DataFrame of
    historical observations, the periods to produce thresholds for, and the
    strategy's validated params model, and returns one row per
    ``(period_id, location, line)``.
    """

    id: str = ""
    name: str = ""
    description: str = ""
    params_model: type[ParamsT]

    @abstractmethod
    def compute(
        self,
        historical_observations: pd.DataFrame,
        period_ids: list[str],
        params: ParamsT,
    ) -> pd.DataFrame:
        """Compute thresholds for the requested periods.

        Args:
            historical_observations: DataFrame with columns
                ``[location, time_period, disease_cases]``.
            period_ids: Periods to produce thresholds for (e.g. ``["2024-01"]``).
            params: Validated instance of :attr:`params_model`.

        Returns:
            DataFrame with columns ``[period_id, location, line, threshold]`` —
            one row per ``(period_id, location, line)``, where ``line`` is the
            zero-based index into the requested line parameter list.
        """


def align_seasonal_to_periods(per_season: pd.DataFrame, period_ids: list[str]) -> pd.DataFrame:
    """Merge per-season threshold lines onto the requested periods.

    Args:
        per_season: DataFrame with columns ``[location, month|week, line, threshold]``.
        period_ids: Periods to produce thresholds for.

    Returns:
        DataFrame with columns ``[period_id, location, line, threshold]``.
    """
    import pandas as pd

    from chap_core.time_period.vectorized import season_column

    requested = pd.DataFrame({"period_id": period_ids})
    season, buckets = season_column(requested["period_id"])
    if season not in per_season.columns:
        raise ValueError(f"period_ids are {season}-based but the dataset's time periods have a different frequency")
    requested[season] = buckets
    merged = requested.merge(per_season, on=season)
    return merged[["period_id", "location", "line", "threshold"]]
