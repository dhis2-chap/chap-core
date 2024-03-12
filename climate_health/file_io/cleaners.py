import numpy as np
import pandas as pd

from climate_health.datatypes import ClimateHealthData
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
from climate_health.time_period.dataclasses import Month


def hydromet(filename):
    df = pd.read_csv(filename)
    df = df.sort_values(by=['micro_code', 'year', 'month'])
    grouped = df.groupby('micro_name_ibge')

    data_dict = {}
    for name, group in grouped:
        period = Month(group['year'], group['month'])
        tmax = group['tmax'].values
        tmin = group['tmin'].values
        tmean = (tmax + tmin) / 2
        data_dict[name] = ClimateHealthData(period, np.zeros_like(tmean), tmean, group['dengue_cases'].values)
    return SpatioTemporalDict(data_dict)




