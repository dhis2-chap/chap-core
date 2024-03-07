from bionumpy.util.testing import assert_bnpdataclass_equal

from climate_health.time_period.period_range import period_range
from climate_health.time_period.dataclasses import Month, Day, Year


def test_period_range():
    start = Month.single_entry(2020, 7)
    end = Month.single_entry(2021, 2)
    true_range = Month(month=[7, 8, 9, 10, 11, 0, 1, 2], year=[2020, 2020, 2020, 2020, 2020, 2021, 2021, 2021])
    assert_bnpdataclass_equal(period_range(start, end), true_range)
