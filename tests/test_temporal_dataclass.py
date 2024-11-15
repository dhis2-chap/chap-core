import pickle

import pytest
import numpy as np
from chap_core.api_types import PeriodObservation
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
