"""Observed weather provider: perfect foresight, for diagnostic runs only."""

from __future__ import annotations

from chap_core.assessment.weather_providers import weather_provider
from chap_core.assessment.weather_providers.base import FutureWeatherProviderBase


@weather_provider(
    "observed",
    "Observed weather",
    "The weather actually observed in the forecast window. Look-ahead: results are "
    "not comparable to what a model can achieve in production.",
    leaks_future_data=True,
)
class ObservedWeatherProvider(FutureWeatherProviderBase):
    def get_future_weather(self, historical_data, period_range, future_data=None, params=None):
        if future_data is None:
            raise ValueError(
                "The 'observed' weather provider needs the forecast window's observations, "
                "which are only available when backtesting against historical data. "
                "Use a forecasting provider such as 'climatology' to predict ahead."
            )
        return future_data.remove_field("disease_cases")
