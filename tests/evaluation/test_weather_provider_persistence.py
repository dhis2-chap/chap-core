"""The future-weather provider is recorded in the .nc file and survives a round trip."""

from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.weather_providers import (
    DEFAULT_WEATHER_PROVIDER_ID,
    LEGACY_WEATHER_PROVIDER_ID,
)


def test_provider_round_trips_through_nc_file(backtest, tmp_path):
    evaluation = Evaluation.from_backtest(backtest)
    filepath = tmp_path / "eval.nc"
    evaluation.to_file(filepath=filepath, model_name="TestModel", model_version="1.0.0")

    assert Evaluation.from_file(filepath).future_weather_provider == DEFAULT_WEATHER_PROVIDER_ID


def test_non_default_provider_round_trips(backtest, tmp_path):
    evaluation = Evaluation.from_backtest(backtest)
    filepath = tmp_path / "eval.nc"
    evaluation.to_file(filepath=filepath, model_name="TestModel", model_version="1.0.0")

    loaded = Evaluation.from_file(filepath)
    observed_filepath = tmp_path / "eval_observed.nc"
    Evaluation(
        loaded.to_backtest(),
        future_weather_provider="observed",
    ).to_file(filepath=observed_filepath, model_name="TestModel", model_version="1.0.0")

    assert Evaluation.from_file(observed_filepath).future_weather_provider == "observed"


def test_legacy_file_is_attributed_to_observed_weather(old_backtest_file):
    """Files predating the provider field were run against the real future weather."""
    assert Evaluation.from_file(old_backtest_file).future_weather_provider == LEGACY_WEATHER_PROVIDER_ID
