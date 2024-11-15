from chap_core.climate_data.harmonization import harmonize_health_data_and_polygons
from chap_core.datatypes import FullData, HealthData, HealthPopulationData
from chap_core.geometry import get_area_polygons, Polygons
from chap_core.google_earth_engine.gee_era5 import Era5LandGoogleEarthEngine
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
df = pd.read_csv('/home/knut/Data/ch_data/uganda_weekly_data.csv')
df['location'] = df['organisationunitid']
df['time_period'] = df['periodcode']
df['disease_cases'] = df['033B-CD01a. Malaria (diagnosed)  - Cases']
population_df = pd.read_csv('/home/knut/Data/ch_data/uganda_population.csv')
population_df['location'] = population_df['organisationunitid']
del population_df['organisationunitid']
population_lookup = {(row['location'], int(year_colname)): row[year_colname] for _, row in population_df.iterrows() for
                     year_colname in ['2019', '2020', '2021', '2022', '2023', '2024']}


# population_df = population_df.melt(id_vars=['location'], var_name='time_period', value_name='population')


def clean_week_codes(week_code):
    year, week = week_code.split('W')
    return f'{year}W{int(week):02d}'


def harmonize_climate(df):
    polygons = Polygons.from_file('/home/knut/Data/ch_data/uganda_orgunits.geojson')
    df['time_period'] = df['time_period'].apply(clean_week_codes)
    dataset = DataSet.from_pandas(df, dataclass=HealthPopulationData, fill_missing=True)
    dataset = dataset.interpolate(['population'])
    # try:
    #     climate_data = Era5LandGoogleEarthEngine().get_historical_era5(polygons.feature_collection().model_dump(), dataset.period_range)
    # except Exception as e:
    #     logging.error(f'Failed to fetch climate data: {e}')
    #     raise
    #harmonized = dataset.merge(climate_data, FullData)
    harmonized = harmonize_health_data_and_polygons(dataset, polygons.feature_collection())
    harmonized = DataSet({k: v[:-10] for k, v in harmonized.items()})

    return harmonized


def harmonize_population(df):
    df['population'] = [population_lookup[(row['location'], int(row['time_period'][:4]))] for _, row in df.iterrows()]
    return df


if __name__ == '__main__':
    name_mapping = {row['organisationunitid'].lower(): row['organisationunitname'] for _, row in df.iterrows()}
    df = harmonize_population(df)
    dataset = harmonize_climate(df)
    dataset = DataSet({name_mapping[k]: v for k, v in dataset.items()})
    dataset.to_csv('/home/knut/Data/ch_data/uganda_weekly_data_harmonized_human.csv')

'''
dataset = DataSet.from_csv('/home/knut/Data/ch_data/uganda_weekly_data.csv', FullData)
convert_name = lambda name: ' '.join(name.split()[:-1])
raw_names = dataset.locations()
place_names = list(map(convert_name, raw_names))

polygons_2 = get_area_polygons('uganda', place_names, 2)
'''
