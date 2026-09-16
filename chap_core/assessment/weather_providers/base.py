"""Base class for future-weather providers.

A provider supplies the climate covariates a model needs for the periods it is
forecasting into - data that does not exist yet at prediction time. Subclasses
implement :meth:`get_future_weather`; registration happens via the
:func:`weather_provider` decorator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
    from chap_core.time_period import PeriodRange


class FutureWeatherProviderBase(ABC):
    """Base class for future-weather providers.

    Subclasses implement :meth:`get_future_weather`, which receives the data
    available up to the prediction point and the periods it must fill.

    A provider that needs the observations from the forecast window itself must
    declare ``leaks_future_data = True``; only then does the caller pass
    ``future_data``. A provider leaving the flag at ``False`` is therefore
    structurally unable to look ahead, rather than merely trusted not to.
    """

    id: str = ""
    name: str = ""
    description: str = ""
    leaks_future_data: bool = False

    @abstractmethod
    def get_future_weather(
        self,
        historical_data: DataSet,
        period_range: PeriodRange,
        future_data: DataSet | None = None,
        params: dict | None = None,
    ) -> DataSet:
        """Produce future weather covariates covering ``period_range``.

        Args:
            historical_data: Data available up to the prediction point.
            period_range: The periods to produce covariates for.
            future_data: Observations from the forecast window. Passed only to
                providers declaring ``leaks_future_data``; ``None`` otherwise.
            params: Optional provider-specific parameters.

        Returns:
            DataSet covering ``period_range``, with the target field removed.
        """
