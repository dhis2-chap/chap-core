import pytest
import numpy as np
from climate_health.api_types import PeriodObservation
from climate_health.time_period import PeriodRange
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict


class HealthObservation(PeriodObservation):
    disease_cases: int


@pytest.fixture
def observations():
    return {'Oslo': [HealthObservation(time_period='2020-01', disease_cases=10),
                     HealthObservation(time_period='2020-02', disease_cases=20)]}


def test_from_period_observations(observations):
    d = SpatioTemporalDict.from_period_observations(observations)
    assert d.locations() == {'Oslo'}
    assert np.all(d['Oslo'].time_period == PeriodRange.from_strings(['2020-01', '2020-02']))