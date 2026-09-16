import pytest

from chap_core.time_period import Month
from chap_core.assessment.dataset_splitting import (
    split_test_train_on_period,
    train_test_split,
    get_split_points_for_period_range,
    train_test_generator,
)
from chap_core.time_period import PeriodRange
from chap_core.assessment.weather_providers import (
    DEFAULT_WEATHER_PROVIDER_ID,
    get_future_weather,
    resolve_weather_provider,
)
from .data_fixtures import full_data


def test_train_test_split(full_data):
    test_start = Month(2012, 7)
    train_data, test_data = train_test_split(full_data, test_start)
    train_table, test_table = (data.get_location("oslo") for data in (train_data, test_data))
    assert len(train_table) == 6
    assert len(test_table) == 6


def test_split_test_train_on_period(full_data):
    split_points = [Month(2012, 3), Month(2012, 7)]
    true_lens = (2, 6)
    for (train_data, test_data), true_len in zip(split_test_train_on_period(full_data, split_points), true_lens):
        train_table, test_table = (data.get_location("oslo") for data in (train_data, test_data))
        assert len(train_table) == true_len
        assert len(test_table) == 12 - true_len


def test_get_split_points_for_period_range():
    period_range = PeriodRange.from_time_periods(Month(2012, 1), Month(2012, 12))
    split_points = get_split_points_for_period_range(1, period_range, start_offset=3)
    assert split_points == [Month(2012, 8)]


def test_train_test_generator(full_data):
    print(full_data)
    train_data, test_pairs_iter = train_test_generator(full_data, prediction_length=3, n_test_sets=2)
    test_pairs = list(test_pairs_iter)
    assert len(test_pairs) == 2
    assert all(len(pair[1].period_range) == 3 for pair in test_pairs)
    assert all(test_pairs[-1][1].period_range == full_data.period_range[-3:])


def test_train_test_generator_defaults_to_non_leaking_provider(full_data):
    """The default must route through a provider that never sees the forecast window."""
    assert resolve_weather_provider(DEFAULT_WEATHER_PROVIDER_ID).leaks_future_data is False

    _, pairs = train_test_generator(full_data, prediction_length=3, n_test_sets=1)
    historic, masked_future, _ = next(iter(pairs))
    expected = get_future_weather(DEFAULT_WEATHER_PROVIDER_ID, historic, masked_future.period_range)
    for location in full_data.keys():
        assert list(masked_future[location].rainfall) == list(expected[location].rainfall)


def test_train_test_generator_observed_provider_returns_true_weather(full_data):
    _, pairs = train_test_generator(full_data, prediction_length=3, n_test_sets=1, future_weather_provider="observed")
    _, masked_future, future_truth = next(iter(pairs))
    for location in full_data.keys():
        assert list(masked_future[location].rainfall) == list(future_truth[location].rainfall)


def test_train_test_generator_rejects_unknown_provider(full_data):
    with pytest.raises(ValueError, match="Unknown future weather provider"):
        _, pairs = train_test_generator(
            full_data, prediction_length=3, n_test_sets=1, future_weather_provider="no_such_provider"
        )
        next(iter(pairs))
