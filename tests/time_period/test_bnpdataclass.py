import pytest
from bionumpy.bnpdataclass import bnpdataclass
from chap_core.time_period import Year, Day, PeriodRange


@bnpdataclass
class TestClass:
    period: Year
    cases: int


def test_year():
    years = PeriodRange.from_strings(['2013', '2014', '2015'])
    assert years.year[0] == 2013

@pytest.mark.xfail
def test_indataclass():
    year = Year([2015, 2014, 2013])
    test = TestClass(year, [1, 2, 3])
    assert test.period.year[0] == 2015
    assert test.sort_by("period").period.year[0] == 2013

@pytest.mark.xfail
def test_argsort():
    year = Year([2015, 2014, 2013])
    assert year.argsort()[0] == 2
    assert year.argsort()[1] == 1
    assert year.argsort()[2] == 0

@pytest.mark.xfail
def test_argsort_days():
    days = Day([2015, 2014, 2013], [1, 2, 3], [1, 2, 3])
    assert days.argsort()[0] == 2
    assert days.argsort()[1] == 1
    assert days.argsort()[2] == 0
