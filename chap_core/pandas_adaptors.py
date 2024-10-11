import pandas as pd


def get_time_period(df, year_name, month_name=None, day_name=None, week_name=None):
    if month_name is not None:
        assert day_name is None, "Cannot have day and month yet"
        assert week_name is None, "Cannot have week and month yet"
        return [pd.Period(f"{year}-{month}", "M") for year, month in zip(df[year_name], df[month_name])]
    assert False, "Only Monthly data is supported so far"
