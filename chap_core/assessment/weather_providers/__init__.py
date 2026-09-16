"""Future-weather provider plugin system.

Each provider is a single class registered with the :func:`weather_provider`
decorator, mirroring the ``@threshold`` and ``@backtest_plot`` registries. New
providers are added by writing a class and importing its module from
:func:`_discover_providers` - the evaluation and endpoint code never needs
editing.

Providers declare whether they read the forecast window's own observations via
``leaks_future_data``. :func:`get_future_weather` forwards ``future_data`` only
to providers that declare it, so look-ahead is an explicit property of a named
provider instead of an accident of which code path ran.

Only seasonal numeric covariates are handed to a forecasting provider. String
columns and non-seasonal numerics such as ``population`` are split off once here
and carried forward from their last observation, so every provider is a pure
forecaster and none has to re-implement that rule.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

from chap_core.assessment.weather_providers.base import FutureWeatherProviderBase

if TYPE_CHECKING:
    from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
    from chap_core.time_period import PeriodRange

#: Provider used when a caller does not name one.
DEFAULT_WEATHER_PROVIDER_ID = "climatology"

#: Provider that results predate the provider field are attributed to. Evaluations
#: written before providers were recorded ran against the forecast window's own
#: observations, so that is what they are labelled with on read.
LEGACY_WEATHER_PROVIDER_ID = "observed"

#: The field the models forecast; never a covariate to be provided.
TARGET_FIELD = "disease_cases"

#: Numeric covariates that carry no seasonal signal. Regressing them on
#: month-of-year would hand the model a climatological population.
NON_SEASONAL_FIELDS = frozenset({"population"})

# Global registry for future weather providers
_weather_providers_registry: dict[str, type[FutureWeatherProviderBase]] = {}


def weather_provider(provider_id: str, name: str, description: str = "", leaks_future_data: bool = False):
    """Decorator to register a future-weather provider class."""

    def decorator(cls: type[FutureWeatherProviderBase]) -> type[FutureWeatherProviderBase]:
        if not issubclass(cls, FutureWeatherProviderBase):
            raise TypeError(f"{cls.__name__} must inherit from FutureWeatherProviderBase")

        cls.id = provider_id
        cls.name = name
        cls.description = description
        cls.leaks_future_data = leaks_future_data

        _weather_providers_registry[provider_id] = cls
        return cls

    return decorator


def list_weather_providers() -> list[dict]:
    return [
        {
            "id": cls.id,
            "name": cls.name,
            "description": cls.description,
            "leaks_future_data": cls.leaks_future_data,
        }
        for cls in _weather_providers_registry.values()
    ]


def resolve_weather_provider(provider_id: str) -> type[FutureWeatherProviderBase]:
    """Look up ``provider_id``, raising a ValueError listing the alternatives."""
    provider_cls = _weather_providers_registry.get(provider_id)
    if provider_cls is None:
        available = ", ".join(_weather_providers_registry)
        raise ValueError(f"Unknown future weather provider: {provider_id!r}. Available: {available}")
    return provider_cls


def _carried_forward_fields(historical_data: DataSet) -> list[str]:
    sample = next(iter(historical_data.values()))
    return [
        field.name
        for field in dataclasses.fields(sample)
        if field.name != "time_period"
        and (field.name in NON_SEASONAL_FIELDS or getattr(sample, field.name).dtype.kind not in ("f", "i"))
    ]


def get_future_weather(
    provider_id: str,
    historical_data: DataSet,
    period_range: PeriodRange,
    future_data: DataSet | None = None,
    params: dict | None = None,
) -> DataSet:
    """Produce future weather for ``period_range`` using the named provider.

    ``future_data`` is forwarded only when the provider declares
    ``leaks_future_data``, so a non-leaking provider never sees the forecast
    window's observations even if the caller has them.

    A forecasting provider receives only the seasonal numeric covariates. The
    remaining columns are carried forward from their last observation and
    re-attached here, so the result has every field of the input except the target.
    """
    provider_cls = resolve_weather_provider(provider_id)
    provider = provider_cls()
    if provider_cls.leaks_future_data:
        return provider.get_future_weather(historical_data, period_range, future_data=future_data, params=params)

    covariates = historical_data.remove_field(TARGET_FIELD)
    carried = _carried_forward_fields(covariates)
    forecastable = covariates
    for name in carried:
        forecastable = forecastable.remove_field(name)
    predicted = provider.get_future_weather(forecastable, period_range, params=params)
    if not carried:
        return predicted

    # Repeat the final observation by index so the column keeps its original
    # container type (bionumpy encodes string columns as ragged arrays).
    repeat = np.zeros(len(period_range), dtype=int)
    prediction_dict = {}
    for location, data in covariates.items():
        fields = {name: getattr(predicted[location], name) for name in predicted.field_names()}
        fields |= {name: getattr(data, name)[-1:][repeat] for name in carried}
        prediction_dict[location] = data.__class__(period_range, **fields)
    return covariates.__class__(prediction_dict)  # type: ignore[no-any-return]


def _discover_providers():
    from chap_core.assessment.weather_providers import (
        climatology,
        damped_persistence,
        mstl_arima,
        observed,
    )


_discover_providers()
