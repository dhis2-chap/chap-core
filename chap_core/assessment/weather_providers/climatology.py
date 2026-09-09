"""Seasonal climatology provider: the historical seasonal signal, projected forward."""

from __future__ import annotations

from chap_core.assessment.weather_providers import weather_provider
from chap_core.assessment.weather_providers.base import FutureWeatherProviderBase


@weather_provider(
    "climatology",
    "Seasonal climatology",
    "Per-location regression on month- or week-of-year, fitted on the historical data.",
)
class ClimatologyWeatherProvider(FutureWeatherProviderBase):
    def get_future_weather(self, historical_data, period_range, future_data=None, params=None):
        # Imported lazily: chap_core.climate_predictor pulls in sklearn, which
        # would otherwise be paid on every import of the registry.
        from chap_core.climate_predictor import get_climate_predictor

        return get_climate_predictor(historical_data).predict(period_range)
