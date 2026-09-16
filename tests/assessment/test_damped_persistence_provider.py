"""Tests for the damped persistence future-weather provider."""

import numpy as np
import pytest

from chap_core.assessment.weather_providers import get_future_weather, resolve_weather_provider

from ..data_fixtures import full_data, multi_year_climate_health_data  # noqa: F401


def test_provider_is_registered_and_does_not_leak():
    assert resolve_weather_provider("damped_persistence").leaks_future_data is False


def test_forecast_covers_requested_locations_and_periods(full_data):  # noqa: F811
    periods = full_data.period_range[-3:]

    result = get_future_weather("damped_persistence", full_data, periods)

    assert set(result.keys()) == set(full_data.keys())
    for data in result.values():
        assert list(map(str, data.time_period)) == list(map(str, periods))


def test_zero_damping_reproduces_climatology(multi_year_climate_health_data):  # noqa: F811
    """With no anomaly carried forward the forecast is exactly the climatology."""
    history = multi_year_climate_health_data
    periods = history.period_range[-3:]

    climatology = get_future_weather("climatology", history, periods)
    damped = get_future_weather("damped_persistence", history, periods, params={"damping": 0.0})

    for location in history.keys():
        assert np.allclose(
            np.asarray(damped[location].rainfall, dtype=float),
            np.asarray(climatology[location].rainfall, dtype=float),
        )


def test_anomaly_decays_toward_climatology_with_horizon(multi_year_climate_health_data):  # noqa: F811
    """The whole point of damping: further out, the forecast reverts to climatology."""
    history = multi_year_climate_health_data
    periods = history.period_range[-6:]

    climatology = get_future_weather("climatology", history, periods)
    damped = get_future_weather("damped_persistence", history, periods, params={"damping": 0.5})

    for location in history.keys():
        gap = np.abs(
            np.asarray(damped[location].rainfall, dtype=float) - np.asarray(climatology[location].rainfall, dtype=float)
        )
        assert np.all(np.diff(gap) <= 1e-12), gap


def test_undamped_persistence_holds_the_anomaly(multi_year_climate_health_data):  # noqa: F811
    """With damping 1 the anomaly is carried forward undiminished."""
    history = multi_year_climate_health_data
    periods = history.period_range[-4:]

    climatology = get_future_weather("climatology", history, periods)
    persisted = get_future_weather("damped_persistence", history, periods, params={"damping": 1.0})

    for location in history.keys():
        gap = np.asarray(persisted[location].rainfall, dtype=float) - np.asarray(
            climatology[location].rainfall, dtype=float
        )
        assert np.allclose(gap, gap[0])


@pytest.mark.parametrize("damping", [0.0, 0.3, 1.0])
def test_forecast_is_finite_for_any_damping(full_data, damping):  # noqa: F811
    result = get_future_weather(
        "damped_persistence", full_data, full_data.period_range[-3:], params={"damping": damping}
    )
    for data in result.values():
        assert np.isfinite(np.asarray(data.rainfall, dtype=float)).all()
