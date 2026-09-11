"""Base class for outbreak models.

A model turns history -- and, for predictive models, forecast samples -- into an
alert probability per ``(location, target period, horizon_distance)``. The
threshold ("epidemic channel") comes from a registered threshold strategy, so an
outbreak model composes a strategy with a way of reading a signal against it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import pandas as pd

from chap_core.assessment.thresholds.params import SeasonalParams
from chap_core.assessment.thresholds.seasonal import SeasonalThresholdStrategy
from chap_core.time_period import TimePeriod

if TYPE_CHECKING:
    from chap_core.assessment.thresholds.base import ThresholdStrategyBase
    from chap_core.assessment.thresholds.params import ThresholdParamsBase

PROBABILITY_COLUMNS = ["location", "time_period", "horizon_distance", "threshold", "probability"]


def restrict_to_origin(historical_observations: pd.DataFrame, origin_period: str | None) -> pd.DataFrame:
    """Drop observations after ``origin_period``.

    Standing at an origin period, a model may only use what was observable then.
    Applying this to the threshold baseline is what keeps a period from raising
    the very channel it is about to be compared against. ``None`` disables the
    restriction and uses every observation supplied.
    """
    if origin_period is None:
        return historical_observations
    origin = TimePeriod.parse(str(origin_period))
    keep = historical_observations["time_period"].astype(str).map(lambda p: TimePeriod.parse(p) <= origin)
    return historical_observations[keep]


class OutbreakModelBase(ABC):
    """Base class for outbreak models.

    Subclasses implement :meth:`alert_probabilities`. Registration happens via
    the :func:`~chap_core.assessment.outbreak.outbreak_model` decorator.
    """

    id: str = ""
    name: str = ""
    description: str = ""

    def __init__(
        self,
        threshold_strategy: ThresholdStrategyBase | None = None,
        threshold_params: ThresholdParamsBase | None = None,
    ):
        """Compose a threshold strategy with this model's way of reading a signal.

        Defaults to the seasonal mean + 2*std channel, which is what the outbreak
        metrics have always scored against.
        """
        strategy: ThresholdStrategyBase[Any] = threshold_strategy or SeasonalThresholdStrategy()
        params: ThresholdParamsBase = threshold_params or SeasonalParams(type="seasonal")
        if not isinstance(params, strategy.params_model):
            raise ValueError(
                f"{type(strategy).__name__} takes {strategy.params_model.__name__} params, got {type(params).__name__}"
            )
        self.threshold_strategy = strategy
        self.threshold_params = params
        if len(self.threshold_params.lines) != 1:
            raise ValueError(
                f"An outbreak model needs exactly one threshold line, got {len(self.threshold_params.lines)}. "
                "Multi-line params draw a channel band; an alert is one decision against one line."
            )

    def thresholds(
        self,
        historical_observations: pd.DataFrame,
        period_ids: list[str],
        origin_period: str | None = None,
    ) -> pd.DataFrame:
        """Compute the epidemic channel for each requested period.

        Args:
            historical_observations: Columns ``[location, time_period, disease_cases]``.
            period_ids: Periods to produce a threshold for.
            origin_period: When given, the baseline uses only observations at or
                before this period. See :func:`restrict_to_origin`.

        Returns:
            Columns ``[time_period, location, threshold]``.
        """
        baseline = restrict_to_origin(historical_observations, origin_period)
        if baseline.empty:
            return pd.DataFrame(columns=["time_period", "location", "threshold"])
        lines = self.threshold_strategy.compute(baseline, period_ids, self.threshold_params)
        single = lines[lines["line"] == 0].drop(columns=["line"])
        return single.rename(columns={"period_id": "time_period"})

    @abstractmethod
    def alert_probabilities(
        self,
        historical_observations: pd.DataFrame,
        target_periods: list[str],
        *,
        origin_period: str | None = None,
        forecasts: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Probability that each location breaches its channel in each target period.

        Args:
            historical_observations: Columns ``[location, time_period, disease_cases]``.
            target_periods: Periods to produce a probability for.
            origin_period: The period the model stands at. Bounds the threshold
                baseline for every model, and names the observation a
                persistence model reads. Required by models that read it.
            forecasts: Columns ``[location, time_period, horizon_distance, sample,
                forecast]``. Required by predictive models, unused by others.

        Returns:
            Columns ``[location, time_period, horizon_distance, threshold, probability]``,
            one row per scored cell. Cells whose threshold cannot be computed are
            omitted.
        """
