import tempfile

import numpy as np

from climate_health.datatypes import ClimateHealthTimeSeries, ClimateData
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
from climate_health.time_period import Month, Year, PeriodRange
from .data_fixtures import full_data
from tempfile import NamedTemporaryFile


def test_to_from_csv(full_data):
    with NamedTemporaryFile() as f:
        full_data.to_csv(f.name)
        full_data2 = full_data.from_csv(f.name, ClimateHealthTimeSeries)
        assert len(full_data2.data()) == len(full_data.data())


def test_climate_data_to_from_csv():
    # just tests that nothing crashes
    future_weather = SpatioTemporalDict(
        {
            "location": ClimateData(
                PeriodRange.from_strings(["2001", "2002", "2003"]),
                np.array([0, 1, 2]),
                np.array([0, 1, 2]),
                np.array([0, 1, 2])
            )
        })

    with tempfile.NamedTemporaryFile() as f:
        future_weather.to_csv(f.name)
        future_weather2 = SpatioTemporalDict.from_csv(f.name, ClimateData)
