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


def test_from_backtest_reads_the_provider_off_the_row(backtest):
    """An `observed` backtest must not convert into a `climatology` evaluation."""
    backtest.future_weather_provider = "observed"

    assert Evaluation.from_backtest(backtest).future_weather_provider == "observed"


def test_from_backtest_exports_the_row_provider(backtest, tmp_path):
    backtest.future_weather_provider = "observed"
    filepath = tmp_path / "eval.nc"

    Evaluation.from_backtest(backtest).to_file(filepath=filepath, model_name="TestModel", model_version="1.0.0")

    assert Evaluation.from_file(filepath).future_weather_provider == "observed"


def test_from_file_sets_the_provider_on_the_backtest(backtest, tmp_path):
    """to_backtest() on a loaded evaluation must carry the provider it was read with."""
    backtest.future_weather_provider = "observed"
    filepath = tmp_path / "eval.nc"
    Evaluation.from_backtest(backtest).to_file(filepath=filepath, model_name="TestModel", model_version="1.0.0")

    loaded = Evaluation.from_file(filepath)

    assert loaded.to_backtest().future_weather_provider == "observed"


def test_constructor_override_lands_on_the_backtest_row(backtest):
    """The row is the single source of truth; an override must be written to it,
    not held beside it, or a to_backtest()/from_backtest() round trip reverts it."""
    evaluation = Evaluation(backtest, future_weather_provider="observed")

    assert evaluation.to_backtest().future_weather_provider == "observed"
    assert Evaluation.from_backtest(evaluation.to_backtest()).future_weather_provider == "observed"


def test_no_override_keeps_the_row_value(backtest):
    backtest.future_weather_provider = "observed"

    assert Evaluation(backtest).future_weather_provider == "observed"
    assert Evaluation(backtest).to_backtest().future_weather_provider == "observed"
