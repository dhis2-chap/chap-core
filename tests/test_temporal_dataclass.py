import pickle

import pytest
import numpy as np
from chap_core.api_types import PeriodObservation
from chap_core.datatypes import FullData, SamplesWithTruth
from chap_core.time_period import PeriodRange
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


class HealthObservation(PeriodObservation):
    disease_cases: int


@pytest.fixture
def observations():
    return {
        "Oslo": [
            HealthObservation(time_period="2020-01", disease_cases=10),
            HealthObservation(time_period="2020-02", disease_cases=20),
        ]
    }


def test_from_period_observations(observations):
    d = DataSet.from_period_observations(observations)
    assert d.locations() == {"Oslo"}
    assert np.all(
        d["Oslo"].time_period == PeriodRange.from_strings(["2020-01", "2020-02"])
    )


@pytest.fixture()
def vietnam_dataset(data_path):
    dataset = DataSet.from_csv(data_path / "vietnam_monthly.csv", FullData)
    return dataset

def test_aggregate_to_parent(vietnam_dataset):
    #aggregated = vietnam_dataset.aggregate_to_parent()
    samples_with_truth = DataSet(
        {location: SamplesWithTruth(time_period=data.time_period,
                                   samples= np.ones((len(data.time_period), 100)),
                                   disease_cases = data.disease_cases)
         for location, data in vietnam_dataset.items()})
    a = samples_with_truth.aggregate_to_parent(field_name='samples')
    print(a)




