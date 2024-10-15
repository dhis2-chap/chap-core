import os
import pickle

import numpy as np
import pandas as pd

from chap_core.data.open_dengue import OpenDengueDataSet
from chap_core.datatypes import ClimateHealthData, FullData
from chap_core.geometry import get_area_polygons, normalize_name
from chap_core.fetch import gee_era5
from chap_core.google_earth_engine.gee_raw import load_credentials
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

country_name = 'NICARAGUA'
level = 1
tmp_name = '{country_name}_era5_data_adm{level}.pkl'
if not os.path.exists(tmp_name):
    base_df = OpenDengueDataSet().as_dataset(country_name, spatial_resolution=f'Admin{level}',
                                             temporal_resolution='Week')
    base_df.to_csv(f'{country_name.lower()}_weekly_cases.csv', index=False)
    if False:
        polygons = get_area_polygons(country_name, base_df['location'].unique(), 2)
        credentials = load_credentials()
        first_week, last_week = min(base_df['time_period']), max(base_df['time_period'])
        data = gee_era5(credentials, polygons, first_week, last_week, ['temperature_2m', 'total_precipitation_sum'])
        pickle.dump(data, open(tmp_name, 'wb'))
else:
    entries = pickle.load(open(tmp_name, 'rb'))
    df = pd.DataFrame([entry.model_dump() for entry in entries])
    df = df.pivot(index=('period', 'location'), columns='band', values='value')
    df = df.reset_index()
    # only keep renamed columns
    df = df.rename(columns={'location': 'location',
                            'period': 'time_period',
                            'temperature_2m': 'mean_temperature',
                            'total_precipitation_sum': 'rainfall'})
    df = df[['location', 'time_period', 'mean_temperature', 'rainfall']]
    df['location'] =[normalize_name(location) for location in df['location']]
    climate_locations = set(df['location'])
    base_df = OpenDengueDataSet().as_dataset(country_name, spatial_resolution=f'Admin{level}',

                                             temporal_resolution='Week')
    base_df['location'] = [normalize_name(location) for location in base_df['location']]
    base_df = base_df[['location', 'time_period', 'disease_cases']]
    base_df = base_df[base_df['location'].isin(climate_locations)]
    # join on location and period, fill missing values with na
    joined = pd.merge(df, base_df, on=['location', 'time_period'], how='outer')
    joined = joined[np.array(['Sun' in period for period in joined['time_period']])]
    joined['population'] = 100000+np.random.randint(1000, size=len(joined))
    ds = DataSet.from_pandas(joined, dataclass=FullData, fill_missing=True)
    ds.to_csv(f'~/Data/ch_data/{country_name.lower()}_weekly_data.csv')
