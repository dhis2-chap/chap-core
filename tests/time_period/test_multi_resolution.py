import numpy as np
from bionumpy.util.testing import assert_bnpdataclass_equal
from numpy.testing import assert_array_equal

from climate_health.time_period.dataclasses import Day, Month
from climate_health.time_period.multi_resolution import pack_to_period
from climate_health.time_period.period_range import period_range


def test_pack_to_period():
    day_range = period_range(Day.single_entry(2020, 0, 0), Day.single_entry(2020, 1, 28))
    month_range = period_range(Month.single_entry(2020, 0), Month.single_entry(2020, 1))
    data = np.arange(31+29)*2
    new_index, new_data = pack_to_period(day_range, data, Month)
    assert_bnpdataclass_equal(new_index, month_range)
    assert_array_equal(new_data.ravel(), data)
    print(new_data.lengths)
    assert tuple(new_data.lengths) == (31, 29)