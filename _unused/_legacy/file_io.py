from typing import List

import chap_core.time_period.dataclasses as dc
from chap_core.time_period import TimePeriod, Month, Day, Year


def parse_period_string(time_string: str) -> TimePeriod:
    period = TimePeriod.parse(time_string)
    return period


def parse_periods_strings(time_strings: List[str]) -> dc.Period:
    periods = [parse_period_string(time_string) for time_string in time_strings]
    if not periods:
        return dc.Period.empty()
    t = type(periods[0])
    assert all(type(period) is t for period in periods), periods

    if t == Year:
        return dc.Year([period.year for period in periods])
    if t == Month:
        return dc.Month([period.year for period in periods], [period.month for period in periods])
    elif t == Day:
        return dc.Day(
            [period.year for period in periods],
            [period.month for period in periods],
            [period.day for period in periods],
        )
