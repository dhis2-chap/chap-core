import json
from typing import Union

import numpy as np
import pandas as pd

from chap_core.datatypes import FullData, HealthData, HealthPopulationData
from chap_core.geometry import get_area_polygons, normalize_name
from chap_core.google_earth_engine.gee_era5 import Era5LandGoogleEarthEngine
from chap_core.google_earth_engine.gee_raw import load_credentials, fetch_era5_data_generator, ERA5Entry
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import TimePeriod
from chap_core.rest_api_src.worker_functions import harmonize_health_dataset as _harmonize_health_dataset

def harmonize_health_dataset(dataset: DataSet[HealthData], country_name, get_climate=True) -> DataSet[FullData]:
    location_names = list(dataset.locations())
    polygons = get_area_polygons(country_name, location_names, 2)
    dataset.set_polygons(polygons)
    return _harmonize_health_dataset(dataset)
    #return harmonize_health_data_and_polygons(dataset, polygons)


def read_broken_json(filename):
    with open(filename) as f:
        data = f.read()
    parts = data[1:-1].split('}{')
    elements =  [json.loads('{' + part + '}') for part in parts]
    return elements


def harmonize_health_data_and_polygons(dataset: Union[HealthData, HealthPopulationData], polygons, cached=False) -> DataSet[FullData]:
    base_df = dataset.to_pandas()
    base_df['time_period'] = [TimePeriod.from_pandas(period).id for period in base_df['time_period']]
    if 'population' not in base_df.columns:
        base_df['population'] = 100000 + np.random.randint(1000, size=len(base_df))

    credentials = load_credentials()
    era5 = Era5LandGoogleEarthEngine()
    data = era5.get_historical_era5(polygons, dataset.period_range)
    first_week, last_week = min(base_df['time_period']), max(base_df['time_period'])
    data = []
    if not cached:
        with open('tmp.json', 'w') as f:
            for entry in fetch_era5_data_generator(credentials, polygons, first_week, last_week, ['temperature_2m', 'total_precipitation_sum']):
                json_data = entry.model_dump()
                json.dump(json_data, f)
                data.append(entry)
    else:
        data = read_broken_json('tmp.json')
        data = [ERA5Entry(**entry) for entry in data]
    entries = data
    df = pd.DataFrame([entry.model_dump() for entry in entries])
    df = df.pivot(index=('period', 'location'), columns='band', values='value')
    df = df.reset_index()
    df = df.rename(columns={'location': 'location',
                            'period': 'time_period',
                            'temperature_2m': 'mean_temperature',
                            'total_precipitation_sum': 'rainfall'})
    df = df[['location', 'time_period', 'mean_temperature', 'rainfall']]
    df['location'] = [normalize_name(location) for location in df['location']]
    climate_locations = set(df['location'])
    base_df['location'] = [normalize_name(location) for location in base_df['location']]
    base_df = base_df[base_df['location'].isin(climate_locations)]
    joined = pd.merge(df, base_df, on=['location', 'time_period'], how='outer')
    ds = DataSet.from_pandas(joined, dataclass=FullData, fill_missing=True)
    return ds
