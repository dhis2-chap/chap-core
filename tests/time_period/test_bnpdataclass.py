from bionumpy.bnpdataclass import bnpdataclass
from climate_health.time_period.dataclasses import Year

@bnpdataclass
class TestClass:
    period: Year
    cases: int

def test_year():
    years = Year([2013, 2014, 2015])
    assert years.year[0] == 2013



def test_indataclass():
    year = Year([2013, 2014, 2015])
    test = TestClass(year, [1, 2, 3])
    assert test.period.year[0] == 2013





