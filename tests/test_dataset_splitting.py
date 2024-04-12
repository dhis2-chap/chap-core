import pytest
from climate_health.time_period import Month
from climate_health.assessment.dataset_splitting import split_test_train_on_period, train_test_split, \
    get_split_points_for_period_range
from climate_health.time_period import PeriodRange
from .data_fixtures import full_data


def test_train_test_split(full_data):
    test_start = Month(2012, 7)
    train_data, test_data = train_test_split(full_data, test_start)
    train_table, test_table = (data.get_location('oslo').data()
                               for data in (train_data, test_data))
    assert len(train_table) == 6
    assert len(test_table) == 6


def test_split_test_train_on_period(full_data):
    split_points = [Month(2012, 3), Month(2012, 7)]
    true_lens = (2, 6)
    for (train_data, test_data), true_len in zip(split_test_train_on_period(full_data, split_points), true_lens):
        train_table, test_table = (data.get_location('oslo').data()
                                   for data in (train_data, test_data))
        assert len(train_table) == true_len
        assert len(test_table) == 12 - true_len



def test_get_split_points_for_period_range():
    period_range = PeriodRange.from_time_periods(Month(2012, 1), Month(2012, 12))
    split_points = get_split_points_for_period_range(1, period_range, start_offset=3)
    assert split_points == [Month(2012, 8)]

