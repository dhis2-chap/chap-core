from typing import List

import chap_core.time_period.dataclasses as dc
from chap_core.time_period import TimePeriod, Month, Day, Year


def parse_period_string(time_string: str) -> TimePeriod:
    period = TimePeriod.parse(time_string)
    return period


def write_time_series_data(data):
    def topandas(self):
        data = pd.DataFrame(
            {
                "time_period": self.time_period.topandas(),
                "rainfall": self.rainfall,
                "mean_temperature": self.mean_temperature,
                "disease_cases": self.disease_cases,
            }
        )
        return data

    to_pandas = topandas

    def to_csv(self, csv_file: str, **kwargs):
        """Write data to a csv file."""
        data = self.to_pandas()
        data.to_csv(csv_file, index=False, **kwargs)


def parse_periods_strings(time_strings: List[str]) -> dc.Period:
    periods = [parse_period_string(time_string) for time_string in time_strings]
    if not periods:
        return dc.Period.empty()
    t = type(periods[0])
    assert all(type(period) == t for period in periods), periods

    if t == Year:
        return dc.Year([period.year for period in periods])
    if t == Month:
        return dc.Month(
            [period.year for period in periods], [period.month for period in periods]
        )
    elif t == Day:
        return dc.Day(
            [period.year for period in periods],
            [period.month for period in periods],
            [period.day for period in periods],
        )
