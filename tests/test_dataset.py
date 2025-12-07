import pandas as pd
import pytest

from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


@pytest.mark.skip
def test_from_samples():
    assert False


def test_from_pandas():
    df = pd.DataFrame(
        {
            "location": ["Oslo", "Oslo", "Bergen", "Bergen"],
            "time_period": ["2020-01", "2020-02", "2020-01", "2020-02"],
            "disease_cases": [10, 20, 30, 40],
        }
    )
    ds = DataSet.from_pandas(df)
    print(ds)


@pytest.fixture
def monthly_dataset():
    df = pd.DataFrame(
        {
            "location": ["Oslo", "Oslo", "Bergen", "Bergen"],
            "time_period": ["2020-01", "2020-02", "2020-01", "2020-02"],
            "disease_cases": [10, 20, 30, 40],
        }
    )
    ds = DataSet.from_pandas(df)
    return ds


@pytest.fixture
def weekly_dataset():
    df = pd.DataFrame(
        {
            "location": ["Oslo", "Oslo", "Bergen", "Bergen"],
            "time_period": ["2020W01", "2020W02", "2020W01", "2020W02"],
            "disease_cases": [10, 20, 30, 40],
        }
    )
    ds = DataSet.from_pandas(df)
    return ds


def test_frequency(monthly_dataset):
    assert monthly_dataset.frequency == "M"


def test_weekly(weekly_dataset):
    assert weekly_dataset.frequency == "W"
