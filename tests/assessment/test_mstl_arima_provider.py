"""Tests for the MSTL + ARIMA future-weather provider."""

import numpy as np
import pytest

from chap_core.assessment.weather_providers import get_future_weather, resolve_weather_provider

from ..data_fixtures import full_data, multi_year_climate_health_data  # noqa: F401


def test_provider_is_registered_and_does_not_leak():
    assert resolve_weather_provider("mstl_arima").leaks_future_data is False


def test_forecast_covers_requested_locations_and_periods(multi_year_climate_health_data):  # noqa: F811
    history = multi_year_climate_health_data.restrict_time_period(
        slice(None, multi_year_climate_health_data.period_range[-3])
    )
    periods = multi_year_climate_health_data.period_range[-2:]

    result = get_future_weather("mstl_arima", history, periods)

    assert set(result.keys()) == set(history.keys())
    for data in result.values():
        assert list(map(str, data.time_period)) == list(map(str, periods))
        assert np.isfinite(data.rainfall).all()


def test_forecast_tracks_the_seasonal_signal(multi_year_climate_health_data):  # noqa: F811
    """On a clean seasonal series the forecast should beat a flat last-value guess."""
    horizon = 6
    full = multi_year_climate_health_data
    history = full.restrict_time_period(slice(None, full.period_range[-(horizon + 1)]))
    periods = full.period_range[-horizon:]

    result = get_future_weather("mstl_arima", history, periods)

    for location in full.keys():
        truth = np.asarray(full[location].rainfall, dtype=float)[-horizon:]
        forecast = np.asarray(result[location].rainfall, dtype=float)
        last_value = float(np.asarray(history[location].rainfall, dtype=float)[-1])
        assert np.abs(forecast - truth).mean() < np.abs(last_value - truth).mean()


def test_short_history_is_rejected_with_a_usable_message(full_data):  # noqa: F811
    with pytest.raises(ValueError, match="more than 2 full seasonal cycles"):
        get_future_weather("mstl_arima", full_data, full_data.period_range[-3:])


def test_exactly_two_cycles_is_rejected_not_crashed(multi_year_climate_health_data):  # noqa: F811
    """MSTL drops periods >= half the series, so 24 monthly points yield no seasonal
    component and statsmodels raises UnboundLocalError from inside. The boundary must
    be refused with an actionable message instead."""
    history = multi_year_climate_health_data
    exactly_two_cycles = history.restrict_time_period(slice(None, history.period_range[23]))
    assert len(exactly_two_cycles.period_range) == 24

    with pytest.raises(ValueError, match="more than 2 full seasonal cycles"):
        get_future_weather("mstl_arima", exactly_two_cycles, history.period_range[24:27])


def test_just_over_two_cycles_is_accepted(multi_year_climate_health_data):  # noqa: F811
    history = multi_year_climate_health_data
    just_over = history.restrict_time_period(slice(None, history.period_range[24]))
    assert len(just_over.period_range) == 25

    result = get_future_weather("mstl_arima", just_over, history.period_range[25:28])

    for data in result.values():
        assert np.isfinite(np.asarray(data.rainfall, dtype=float)).all()


def test_forecast_stays_within_a_plausible_range(multi_year_climate_health_data):  # noqa: F811
    """Near-unit-root models forecast explosive oscillations; they must be rejected.

    On real admin1 data an unguarded AICc search picked such models for a handful
    of series, forecasting rainfall of +-5000 against a historical range of 1-18.
    """
    history = multi_year_climate_health_data
    periods = history.period_range[-3:]

    result = get_future_weather("mstl_arima", history, periods)

    for location in history.keys():
        observed = np.asarray(history[location].rainfall, dtype=float)
        spread = observed.max() - observed.min()
        forecast = np.asarray(result[location].rainfall, dtype=float)
        assert np.abs(forecast - observed.mean()).max() < 3 * spread


def test_explicit_order_skips_the_search(multi_year_climate_health_data):  # noqa: F811
    """A caller-supplied order is used as given, which is much faster."""
    history = multi_year_climate_health_data
    periods = history.period_range[-3:]

    auto = get_future_weather("mstl_arima", history, periods)
    fixed = get_future_weather("mstl_arima", history, periods, params={"order": (0, 0, 0)})

    for location in history.keys():
        assert not np.allclose(
            np.asarray(auto[location].rainfall, dtype=float),
            np.asarray(fixed[location].rainfall, dtype=float),
        )
