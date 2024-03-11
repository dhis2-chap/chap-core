import pytest
from climate_health.time_period import Month
from climate_health.assessment.dataset_splitting import split_test_train_on_period, train_test_split
from .data_fixtures import full_data


def test_split_test_train_on_period(full_data):
    test_start = Month(2012, 7)
    train_data, test_data = train_test_split(full_data, test_start)
    train_table, test_table = (data.get_location('oslo').data()
                               for data in (train_data, test_data))
    assert len(train_table) == 6
    assert len(test_table) == 6
