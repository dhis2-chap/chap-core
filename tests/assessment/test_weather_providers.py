import numpy as np
import pytest

from chap_core.assessment.weather_providers import (
    DEFAULT_WEATHER_PROVIDER_ID,
    get_future_weather,
    list_weather_providers,
    resolve_weather_provider,
    weather_provider,
)
from chap_core.assessment.weather_providers.base import FutureWeatherProviderBase
from chap_core.api_types import BacktestParams
from chap_core.rest_api.data_models import PredictionParams

from ..data_fixtures import full_data, full_data_with_gap, full_data_with_parent, train_data_pop  # noqa: F401


def test_builtin_providers_are_registered():
    ids = {p["id"] for p in list_weather_providers()}
    assert {"climatology", "observed"} <= ids


def test_listing_exposes_id_name_description_and_leak_flag():
    by_id = {p["id"]: p for p in list_weather_providers()}
    assert by_id["climatology"]["leaks_future_data"] is False
    assert by_id["observed"]["leaks_future_data"] is True
    for provider_id in ("climatology", "observed"):
        assert by_id[provider_id]["name"]
        assert by_id[provider_id]["description"]


def test_default_provider_does_not_leak():
    assert resolve_weather_provider(DEFAULT_WEATHER_PROVIDER_ID).leaks_future_data is False


def test_resolve_unknown_provider_lists_alternatives():
    with pytest.raises(ValueError, match="Unknown future weather provider"):
        resolve_weather_provider("no_such_provider")


def test_decorator_rejects_class_not_deriving_from_base():
    with pytest.raises(TypeError, match="must inherit from FutureWeatherProviderBase"):

        @weather_provider("bad_provider", "Bad")
        class NotAProvider:
            pass


@weather_provider("test_recorder_no_leak", "Recorder (no leak)")
class _RecordingProvider(FutureWeatherProviderBase):
    """Records whatever future_data the caller passed it."""

    last_future_data: object = "unset"

    def get_future_weather(self, historical_data, period_range, future_data=None, params=None):
        type(self).last_future_data = future_data
        return historical_data.remove_field("disease_cases")


@weather_provider("test_recorder_leaks", "Recorder (leaks)", leaks_future_data=True)
class _LeakingRecordingProvider(_RecordingProvider):
    pass


def test_future_data_withheld_from_non_leaking_provider(full_data):  # noqa: F811
    get_future_weather("test_recorder_no_leak", full_data, full_data.period_range, future_data=full_data)
    assert _RecordingProvider.last_future_data is None


def test_future_data_passed_to_leaking_provider(full_data):  # noqa: F811
    get_future_weather("test_recorder_leaks", full_data, full_data.period_range, future_data=full_data)
    assert _LeakingRecordingProvider.last_future_data is full_data


def test_observed_provider_returns_the_future_window(full_data):  # noqa: F811
    future = full_data.restrict_time_period(slice(full_data.period_range[-3], None))
    result = get_future_weather("observed", full_data, future.period_range, future_data=future)
    assert "disease_cases" not in result.field_names()
    for location in future.keys():
        assert list(result[location].rainfall) == list(future[location].rainfall)


def test_observed_provider_refuses_to_predict_ahead(full_data):  # noqa: F811
    with pytest.raises(ValueError, match="only available when backtesting"):
        get_future_weather("observed", full_data, full_data.period_range[-3:])


def test_climatology_provider_covers_requested_periods(full_data):  # noqa: F811
    periods = full_data.period_range[-3:]
    result = get_future_weather("climatology", full_data, periods)
    assert set(result.keys()) == set(full_data.keys())
    for data in result.values():
        assert list(map(str, data.time_period)) == list(map(str, periods))


def test_backtest_params_defaults_to_the_default_provider():
    assert BacktestParams().future_weather_provider == DEFAULT_WEATHER_PROVIDER_ID


def test_prediction_params_defaults_to_the_default_provider():
    assert PredictionParams(model_id="m").future_weather_provider == DEFAULT_WEATHER_PROVIDER_ID


@pytest.mark.parametrize(
    "build_params",
    [
        lambda value: BacktestParams(future_weather_provider=value),
        lambda value: PredictionParams(model_id="m", future_weather_provider=value),
    ],
    ids=["backtest_params", "prediction_params"],
)
def test_params_reject_unregistered_provider(build_params):
    with pytest.raises(ValueError, match="Unknown future weather provider"):
        build_params("no_such_provider")


def test_prediction_params_reject_look_ahead_provider():
    """A provider that reads the forecast window cannot predict ahead, so the
    request must fail at validation rather than inside the worker."""
    with pytest.raises(ValueError, match="cannot forecast ahead"):
        PredictionParams(model_id="m", future_weather_provider="observed")


def test_climatology_carries_non_numeric_fields_forward(full_data_with_parent):  # noqa: F811
    """Static string columns (org unit parents, codes) have no seasonal signal to fit."""
    periods = full_data_with_parent.period_range[-3:]

    result = get_future_weather("climatology", full_data_with_parent, periods)

    for location in full_data_with_parent.keys():
        assert [str(value) for value in result[location].parent] == ["norway"] * len(periods)


@pytest.mark.parametrize("provider_id", ["climatology", "damped_persistence"])
def test_forecasting_providers_carry_population_forward(train_data_pop, provider_id):  # noqa: F401, F811
    """Population is not seasonal; regressing it on month-of-year would hand the
    model a climatological population instead of the current one."""
    periods = train_data_pop.period_range[-3:]

    result = get_future_weather(provider_id, train_data_pop, periods)

    for location, data in train_data_pop.items():
        assert list(result[location].population) == [data.population[-1]] * len(periods)
        assert "disease_cases" not in result.field_names()


def test_missing_covariate_values_do_not_break_the_fit(full_data_with_gap):  # noqa: F811
    periods = full_data_with_gap.period_range[-3:]

    result = get_future_weather("climatology", full_data_with_gap, periods)

    for data in result.values():
        assert np.isfinite(np.asarray(data.rainfall, dtype=float)).all()
