from typing import List

from climate_health.time_period.dataclasses import Period
from climate_health.time_period import TimePeriod, Month, Day, Year



def parse_period_string(time_string: str) -> TimePeriod:
    period = TimePeriod.from_string(time_string)
    return period

def parse_period_strings(time_strings: List[str]) -> Period:
    periods = [parse_period_string(time_string) for time_string in time_strings]
    if not periods:
        return Period.empty()
    t = type(periods[0])
    assert all(type(period) == t for period in periods), periods
    if t == Month:
        return Month([period.year for period in periods],
                     [period.month for period in periods])
    elif t == Day:
        return Day([period.year for period in periods],
                   [period.month for period in periods],
                   [period.day for period in periods])

