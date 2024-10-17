from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from chap_core.time_period.date_util_wrapper import (
    TimePeriod,
    TimeStamp,
    delta_month,
    PeriodRange,
    delta_year,
    Month,
    Day,
    Year,
    Week,
)


@pytest.fixture
def period1():
    return TimePeriod.parse("2020-1")


@pytest.fixture
def period2():
    return TimePeriod.parse("2020-2")


@pytest.fixture
def period3():
    return TimePeriod.parse("2021-2")


@pytest.fixture
def edge_timestamps():
    texts = ["2019-12-31", "2020-1-1", "2020-1-31", "2020-2-1"]
    return [TimeStamp.parse(text) for text in texts]


def test_init_with_numbers(period2):
    assert Month(2020, 2) == period2
    assert Day(2020, 2, 3) == Day(datetime(2020, 2, 3))
    assert Year(2020) == TimePeriod.parse("2020")


def test_init_week_with_numbers():
    week = Week(2023, 2)
    assert isinstance(week, Week)
    assert week.start_timestamp == TimeStamp.parse("2023-01-09")
    assert week.to_string() == "2023W2"  # pd.Period('2023-01-09', freq='W-MON')


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


def test_period_id(period1):
    assert period1.id == "202001"
    assert Week(2023, 2).id == "2023W02"
    assert Day(2023, 2, 3).id == "20230203"
    assert Year(2023).id == "2023"


def test_from_id(period1):
    assert TimePeriod.from_id("202001") == period1
    assert TimePeriod.from_id("2023W02") == Week(2023, 2)
    assert TimePeriod.from_id("20230203") == Day(2023, 2, 3)
    assert TimePeriod.from_id("2023") == Year(2023)


@pytest.fixture
def period_range(period1, period3):
    return PeriodRange.from_time_periods(start_period=period1, end_period=period3)

@pytest.fixture()
def weekly_period_range():
    start_period = Week(2020, 1)
    end_period = Week(2020, 3)
    return PeriodRange.from_time_periods(start_period=start_period, end_period=end_period)

def test_weekly_to_pandas(weekly_period_range):
    df = weekly_period_range.topandas()
    print(df)
    pr = PeriodRange.from_pandas(df)
    assert all(pr == weekly_period_range)

def test_period_range(period_range):
    assert len(period_range) == 14


# @pytest.mark.xfail
def test_period_range_slice(period_range):
    assert len(period_range[:1]) == 1
    assert len(period_range[1:]) == len(period_range) - 1
    assert period_range[1:][0] == period_range[1]
    assert len(period_range[1:3]) == 2
    assert len(period_range[1:-2]) == 11
    assert len(period_range[-10:10]) == 6


def test_period_range_iter(period_range, period1, period3):
    assert len(period_range) == 14
    l = list(period_range)
    assert len(l) == 14
    assert l[0] == period1
    assert l[-1] == period3


def test_period_range_eq(period_range, period2):
    mask = period_range == period2
    assert len(mask) == len(period_range)
    assert mask[1]
    assert mask.sum() == 1


def test_period_range_ne(period_range, period2):
    mask = period_range != period2
    assert len(mask) == len(period_range)
    assert not mask[1]
    assert mask.sum() == 13


def test_period_range_lt(period_range, period2):
    mask = period_range < period2
    assert len(mask) == len(period_range)
    assert mask[0]
    assert mask.sum() == 1


def test_period_range_le(period_range, period2):
    mask = period_range <= period2
    assert len(mask) == len(period_range)
    assert mask[0] and mask[1]
    assert mask.sum() == 2


def test_period_range_gt(period_range, period2):
    mask = period_range > period2
    assert len(mask) == len(period_range)
    assert (not mask[0]) and (not mask[1])
    assert mask.sum() == 12


def test_period_range_ge(period_range, period2):
    mask = period_range >= period2
    assert len(mask) == len(period_range)
    assert not mask[0]
    assert mask.sum() == 13


# def period_range_ge():
#    period_range = PeriodRange(start_period=period1, end_period=period3)
def test_topandas(period_range):
    pd_series = period_range.topandas()
    assert pd_series[0] == pd.Period("2020-01")
    assert pd_series[1] == pd.Period("2020-02")
    assert pd_series[13] == pd.Period("2021-02")
    assert len(pd_series) == 14


def test_from_pandas(period_range):
    series = pd.Series(
        [pd.Period("2020-01"), pd.Period("2020-02"), pd.Period("2020-03")]
    )
    period_range = PeriodRange.from_pandas(series)
    assert len(period_range) == 3
    assert period_range[0] == Month(2020, 1)
    assert period_range[1] == Month(2020, 2)
    assert period_range[2] == Month(2020, 3)


def test_from_pandas_inconsecutive(period_range):
    series = pd.Series([pd.Period("2020-01"), pd.Period("2020-03")])
    with pytest.raises(ValueError):
        period_range = PeriodRange.from_pandas(series)


@pytest.mark.parametrize(
    "periods, missing",
    [
        (["2020", "2021", "2023"], [2]),
        (["2020", "2021", "2022", "2023"], []),
        (["2020", "2022", "2023"], [1]),
        (["2020", "2023"], [1, 2]),
        (["2020W1", "2020W2", "2020W4"], [2]),
    ],
)
def test_from_strings_fill_missing(periods, missing):
    period_range, missing_idx = PeriodRange.from_strings(periods, fill_missing=True)
    assert period_range[0] == TimePeriod.parse(periods[0])
    assert period_range[-1] == TimePeriod.parse(periods[-1])

    assert np.all(missing_idx == missing)


def test_searchsorted(period_range, period2):
    array_comparison = np.arange(len(period_range))
    assert period_range.searchsorted(period2) == array_comparison.searchsorted(1)
    assert period_range.searchsorted(
        period2, side="right"
    ) == array_comparison.searchsorted(1, side="right")
    assert period_range.searchsorted(
        period2, side="left"
    ) == array_comparison.searchsorted(1, side="left")


def test_from_start_and_n_periods():
    start_period = pd.Period("2020-01")
    n_periods = 3
    period_range = PeriodRange.from_start_and_n_periods(start_period, n_periods)
    assert len(period_range) == n_periods
    assert period_range[0] == TimePeriod.from_pandas(start_period)
