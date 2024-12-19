from typing import List

import numpy as np
import pytest
from bionumpy.bnpdataclass import bnpdataclass
from bionumpy.util.testing import assert_bnpdataclass_equal
from npstructures import RaggedArray
from numpy.testing import assert_array_equal

from chap_core.datatypes import Location
from chap_core.time_period import Day, Month, PeriodRange
from chap_core.time_period.multi_resolution import pack_to_period



@pytest.mark.skip("legacy")
def test_pack_to_period():
    day_range = PeriodRange.from_time_periods(
        Day.single_entry(2020, 0, 0), Day.single_entry(2020, 1, 28)
    )
    month_range = PeriodRange.from_time_periods(Month.single_entry(2020, 0), Month.single_entry(2020, 1))
    data = np.arange(31 + 29) * 2
    new_index, new_data = pack_to_period(day_range, data, Month)
    assert_bnpdataclass_equal(new_index, month_range)
    assert_array_equal(new_data.ravel(), data)
    print(new_data.lengths)
    assert tuple(new_data.lengths) == (31, 29)


@bnpdataclass
class MultiResolutionClimateData:
    time_period: Month
    temperature: List[float]
    rainfall: List[float]
    humidity: List[float]


@pytest.mark.skip("legacy")
def test_multi_resolution_weather_data(climate_database):
    daily_weather = climate_database.get_data(
        Location(0, 0), Day.single_entry(2020, 0, 0), Day.single_entry(2020, 1, 28)
    )
    time_period, packed = pack_to_period(
        daily_weather.time_period, daily_weather.rainfall, Month
    )
    packed_dc = MultiResolutionClimateData(time_period, packed, packed, packed)
    assert isinstance(packed_dc.time_period, Month)
    assert isinstance(packed_dc.temperature, RaggedArray)
