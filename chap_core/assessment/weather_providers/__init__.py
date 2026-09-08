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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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


def get_weather_providers_registry() -> dict[str, type[FutureWeatherProviderBase]]:
    return _weather_providers_registry.copy()


def get_weather_provider(provider_id: str) -> type[FutureWeatherProviderBase] | None:
    return _weather_providers_registry.get(provider_id)


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
    """
    provider_cls = resolve_weather_provider(provider_id)
    return provider_cls().get_future_weather(
        historical_data,
        period_range,
        future_data=future_data if provider_cls.leaks_future_data else None,
        params=params,
    )


def _discover_providers():
    from chap_core.assessment.weather_providers import (
        climatology,
        damped_persistence,
        mstl_arima,
        observed,
    )


_discover_providers()
