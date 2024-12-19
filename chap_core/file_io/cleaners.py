import numpy as np
import pandas as pd

from chap_core.datatypes import FullData, HealthData
from chap_core.dhis2_interface.periods import convert_time_period_string
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import PeriodRange
#from chap_core.time_period.dataclasses import Month
from chap_core.time_period.date_util_wrapper import Month

def hydromet(filename):
    df = pd.read_csv(filename)
    df = df.sort_values(by=["micro_code", "year", "month"])
    grouped = df.groupby("micro_code")

    data_dict = {}
    for name, group in grouped:
        period = Month(group["year"], group["month"])
        tmax = group["tmax"].values
        tmin = group["tmin"].values
        tmean = (tmax + tmin) / 2
        data_dict[name] = FullData(
            period,
            np.zeros_like(tmean),
            tmean,
            group["dengue_cases"].values,
            group["population"].values,
        )
    return DataSet(data_dict)


def rwanda_data(filename):
    filename = "/home/knut/Downloads/data/Malaria Cases Final.xlsx"
    df = pd.read_excel(filename, sheet_name="Sheet1")
    df.to_csv("/home/knut/Downloads/data/malaria_cases.csv")
    case_names = "Under5_F	Under5_M	5-19_F	5-19_M	20 +_F	20 +_M".split("\t")
    case_names = [name.strip() for name in case_names]
    cases = sum([df[name].values for name in case_names])
    period = [pd.Period(f"{year}-{month}") for year, month in zip(df["Year"], df["Period"])]
    clean_df = pd.DataFrame({"location": df["Sector"], "time_period": period, "disease_cases": cases})
    clean_df.to_csv("/home/knut/Downloads/data/malaria_clean.csv")
    return DataSet.from_pandas(clean_df, dataclass=HealthData)


def laos_data(filename):
    df = pd.read_csv(filename)
    df = df.sort_values(by=["periodid"])
    periods = [convert_time_period_string(str(row)) for row in df["periodid"]]
    print(periods)
    period_range = PeriodRange.from_strings(periods)
    return DataSet({colname: HealthData(period_range, df[colname].values) for colname in df.columns[4:]})
