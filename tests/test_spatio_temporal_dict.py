import tempfile

import numpy as np
import pytest

from chap_core.datatypes import (
    ClimateHealthTimeSeries,
    ClimateData,
    TimeSeriesData,
)
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import Month, PeriodRange
from tempfile import NamedTemporaryFile


def test_to_from_csv(full_data):
    with NamedTemporaryFile() as f:
        full_data.to_csv(f.name)
        full_data2 = full_data.from_csv(f.name, ClimateHealthTimeSeries)
        assert len(full_data2.data()) == len(full_data.data())


def test_climate_data_to_from_csv():
    # just tests that nothing crashes
    future_weather = DataSet(
        {
            "location": ClimateData(
                PeriodRange.from_strings(["2001", "2002", "2003"]),
                np.array([0, 1, 2]),
                np.array([0, 1, 2]),
                np.array([0, 1, 2]),
            )
        }
    )

    with tempfile.NamedTemporaryFile() as f:
        future_weather.to_csv(f.name)
        future_weather2 = DataSet.from_csv(f.name, ClimateData)

@pytest.mark.skip
def test_restrict_on_time_period(train_data_new_period_range):
    period = slice(Month(2012, 1), Month(2012, 4))
    data = train_data_new_period_range.get_location("oslo")
    new_data = data.restrict_time_period(period)
    assert len(new_data.data()) == 4


def test_join_on_time(train_data_new_period_range):
    period1 = slice(Month(2012, 1), Month(2012, 4))
    period2 = slice(Month(2012, 5), Month(2012, 7))

    data1 = train_data_new_period_range.restrict_time_period(period1)
    data2 = train_data_new_period_range.restrict_time_period(period2)

    joined = data1.join_on_time(data2)

    for location in joined.locations():
        period = joined.get_location(location).data().time_period
        assert np.all(
            period == PeriodRange.from_time_periods(Month(2012, 1), Month(2012, 7))
        )


def test_get_location(health_population_data):
    location_data = health_population_data.get_location("FRmrFTE63D0")
    assert isinstance(location_data, TimeSeriesData)


def test_getitem(health_population_data):
    location_data = health_population_data["FRmrFTE63D0"]
    assert isinstance(location_data, TimeSeriesData)
