import pytest

from climate_health.time_period.date_util_wrapper import TimePeriod, TimeStamp, delta_month, PeriodRange, delta_year


@pytest.fixture
def period1():
    return TimePeriod.parse('2020-1')


@pytest.fixture
def period2():
    return TimePeriod.parse('2020-2')


@pytest.fixture
def period3():
    return TimePeriod.parse('2021-2')


@pytest.fixture
def edge_timestamps():
    texts = ['2019-12-31', '2020-1-1', '2020-1-31', '2020-2-1']
    return [TimeStamp.parse(text) for text in texts]


def test_parse(period1):
    assert period1.year == 2020
    assert period1.month == 1


def test_ge(period1, period2):
    assert period2 >= period1
    assert period1 >= period1


def test_le(period1, period2):
    assert not (period2 <= period1)
    assert period1 <= period1


def test_greater_than(period1, period2):
    assert period2 > period1
    assert not period1 > period2


def test_less_than(period1, period2):
    assert period1 < period2
    assert not period2 < period1


def test_compare_period_and_timestamp(period1, edge_timestamps):
    le_comparisons = [period1 <= ts for ts in edge_timestamps]
    assert le_comparisons == [False, True, True, True]
    lt_comparisons = [period1 < ts for ts in edge_timestamps]
    assert lt_comparisons == [False, False, False, True]
    ge_comparisons = [period1 >= ts for ts in edge_timestamps]
    assert ge_comparisons == [True, True, True, False]
    gt_comparisons = [period1 > ts for ts in edge_timestamps]
    assert gt_comparisons == [True, False, False, False]


def test_add_month(period1, period2):
    assert period1 + delta_month == period2


def test_divide_timedelta():
    assert delta_year // delta_month == 12


def test_period_range(period1, period3):
    period_range = PeriodRange(start_period=period1, end_period=period3)
    assert len(period_range) == 14

#def period_range_ge():
#    period_range = PeriodRange(start_period=period1, end_period=period3)
