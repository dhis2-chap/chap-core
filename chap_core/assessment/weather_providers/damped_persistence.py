"""Damped persistence provider.

The forecast is the seasonal climatology plus the current anomaly, damped toward
zero as the horizon grows::

    y(T+h) = climatology(T+h) + phi**h * (y(T) - climatology(T))

With ``phi = 0`` this is exactly the climatology provider; with ``phi = 1`` it is
undamped anomaly persistence. The damping factor defaults to the lag-1
autocorrelation of the location's own anomaly series, which is the optimal
coefficient if anomalies follow an AR(1).

Forecast verification treats plain climatology and persistence as benchmarks
that are too easy - neither uses the observations well, so both flatter whatever
is compared against them. Damped persistence combines the two and is the
reference the sub-seasonal literature recommends instead.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from chap_core.assessment.weather_providers import weather_provider
from chap_core.assessment.weather_providers.base import FutureWeatherProviderBase


def _lag1_autocorrelation(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of ``x``, clipped to [0, 1].

    A negative estimate would make the forecast oscillate about the climatology
    and one above 1 would make it diverge; neither is a sensible damping factor,
    so both are clipped away.
    """
    if len(x) < 3:
        return 0.0
    centred = x - x.mean()
    denominator = float(np.dot(centred, centred))
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(np.dot(centred[:-1], centred[1:]) / denominator, 0.0, 1.0))


@weather_provider(
    "damped_persistence",
    "Damped persistence",
    "Seasonal climatology plus the current anomaly, damped toward zero as the horizon grows. "
    "The reference forecast recommended over plain climatology or persistence at sub-seasonal range.",
)
class DampedPersistenceWeatherProvider(FutureWeatherProviderBase):
    def get_future_weather(self, historical_data, period_range, future_data=None, params=None):
        # Imported lazily: temporal_dataclass imports api_types, which imports this
        # registry for the BacktestParams default, and climate_predictor pulls in sklearn.
        from chap_core.climate_predictor import get_climate_predictor
        from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

        params = params or {}
        damping = params.get("damping")
        n_periods = len(period_range)
        steps = np.arange(1, n_periods + 1)

        predictor = get_climate_predictor(historical_data)
        # The same seasonal fit evaluated over the future window and over the
        # history, so the anomaly is measured against the curve it decays back to.
        climatology_future = predictor.predict(period_range)
        climatology_history = predictor.predict(historical_data.period_range)

        prediction_dict = {}
        for location, data in historical_data.items():
            fields = {}
            for field in dataclasses.fields(data):
                if field.name == "time_period":
                    continue
                future_seasonal = getattr(climatology_future[location], field.name)
                anomaly = np.asarray(getattr(data, field.name), dtype=float) - np.asarray(
                    getattr(climatology_history[location], field.name), dtype=float
                )
                phi = _lag1_autocorrelation(anomaly) if damping is None else float(damping)
                fields[field.name] = np.asarray(future_seasonal, dtype=float) + phi**steps * anomaly[-1]
            prediction_dict[location] = data.__class__(period_range, **fields)
        return DataSet(prediction_dict)
