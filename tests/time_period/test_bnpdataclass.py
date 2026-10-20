from bionumpy.bnpdataclass import bnpdataclass
from chap_core.time_period import Year, PeriodRange


@bnpdataclass
class SampleTimeSeries:
    period: Year
    cases: int


def test_year():
    years = PeriodRange.from_strings(["2013", "2014", "2015"])
    assert years.year[0] == 2013
