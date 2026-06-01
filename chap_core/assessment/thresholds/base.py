"""Base class for threshold (endemic channel) calculation strategies.

A strategy turns a dataset's historical ``disease_cases`` observations into one
threshold value per requested ``(period_id, org_unit)``. Subclasses implement
:meth:`compute`; registration happens via the :func:`threshold` decorator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class ThresholdStrategyBase(ABC):
    """Base class for threshold strategies.

    Subclasses implement :meth:`compute`, which receives a flat DataFrame of
    historical observations and the periods to produce thresholds for, and
    returns one threshold per ``(period_id, org_unit)``.
    """

    id: str = ""
    name: str = ""
    description: str = ""

    @abstractmethod
    def compute(
        self,
        historical_observations: pd.DataFrame,
        period_ids: list[str],
        params: dict | None = None,
    ) -> pd.DataFrame:
        """Compute thresholds for the requested periods.

        Args:
            historical_observations: DataFrame with columns
                ``[location, time_period, disease_cases]``.
            period_ids: Periods to produce a threshold for (e.g. ``["2024-01"]``).
            params: Optional strategy-specific parameters.

        Returns:
            DataFrame with columns ``[period_id, org_unit, threshold]`` — one row
            per ``(period_id, org_unit)``.
        """
