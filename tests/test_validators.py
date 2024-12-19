import numpy as np
import pytest

from chap_core.datatypes import FullData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import PeriodRange
from chap_core.validators import validate_training_data


@pytest.fixture
def small_training_data_months():
    period_range = PeriodRange.from_strings(['2020-01', '2020-02'])
    return zero_data(period_range)


@pytest.fixture
def small_training_data_weeks():
    period_range = PeriodRange.from_strings(['2020W01', '2020W02'])
    return zero_data(period_range)


def zero_data(period_range):
    data = FullData(period_range,
                    np.zeros(len(period_range)),
                    np.zeros(len(period_range)),
                    np.zeros(len(period_range)),
                    np.zeros(len(period_range)))
    return DataSet({'location1': data})


@pytest.fixture()
def big_training_data_months():
    period_range = PeriodRange.from_start_and_n_periods('2020-01', 24)
    return zero_data(period_range)


@pytest.fixture()
def just_too_small_months():
    period_range = PeriodRange.from_start_and_n_periods('2020-01', 23)
    return zero_data(period_range)


@pytest.fixture()
def just_too_small_weeks():
    period_range = PeriodRange.from_start_and_n_periods('2020W03', 103)
    return zero_data(period_range)


def test_validate_training_data(small_training_data_months):
    with pytest.raises(ValueError):
        validate_training_data(small_training_data_months, None)


def test_validate_training_data(small_training_data_weeks):
    with pytest.raises(ValueError):
        validate_training_data(small_training_data_weeks, None)


def test_validate_training_data(just_too_small_months):
    with pytest.raises(ValueError):
        validate_training_data(just_too_small_months, None)


def test_validate_training_data(just_too_small_weeks):
    with pytest.raises(ValueError):
        validate_training_data(just_too_small_weeks, None)


def test_validate_training_data(big_training_data_months):
    validate_training_data(big_training_data_months, None)
