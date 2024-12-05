from typing import Literal
from collections import Counter

import numpy as np
from dateutil.parser import parse
import pandas as pd
import pooch

from chap_core.time_period import Week


class OpenDengueDataSet:
    data_path = 'https://github.com/OpenDengue/master-repo/raw/main/data/releases/V1.2.2/Temporal_extract_V1_2_2.zip'
    def __init__(self):
        self._filename = pooch.retrieve(self.data_path, None)

    def subset(self, country_name: str, spatial_resolution: Literal['Admin1', 'Admin2']='Admin1', temporal_resolution='Week'):
        country_name = country_name.upper()
        df = pd.read_csv(self._filename, compression='zip')
        df = df[df['adm_0_name'] == country_name.upper()]
        df = df[df['T_res'] == temporal_resolution.capitalize()]
        df = df[df['S_res'] == spatial_resolution.capitalize()]
        return df

    def as_dataset(self, country_name: str, spatial_resolution: Literal['Admin1', 'Admin2']='Admin1', temporal_resolution='Week'):
        subset = self.subset(country_name, spatial_resolution, temporal_resolution)
        if temporal_resolution == 'Week':
            dates = [parse(date) for date in subset['calendar_start_date']]
            weekdays = [date.weekday() for date in dates]
            most_common_weekday = Counter(weekdays).most_common(1)[0][0]
            mask = np.array([date.weekday() == most_common_weekday for date in dates])
            subset = subset[mask]
            subset['time_period'] = [Week(parse(date)).id for date in subset['calendar_start_date']]
        elif temporal_resolution == 'Month':
            dates = [parse(date).strftime('%Y-%m') for date in subset['calendar_start_date']]
            subset['time_period'] = dates


        if spatial_resolution == 'Admin1':
            location_column = 'adm_1_name'
        else:
            subset['location'] = subset['adm_1_name'] + '_' + subset['adm_2_name']
            location_column = 'location'
        #location_column = 'adm_1_name' if spatial_resolution == 'Admin1' else 'adm_2_name'
        s = subset.rename(
            columns={location_column: 'location', 'time_period': 'time_period', 'dengue_total': 'disease_cases'})
        assert 'disease_cases' in s.columns, f'No disease_cases column in {s.columns}'
        return s[['location', 'time_period', 'disease_cases']]

