from unidecode import unidecode
import os
from typing import re
import pandas as pd
from pydantic_geojson import FeatureCollectionModel, FeatureModel

from chap_core.datatypes import HealthPopulationData, FullData, SimpleClimateData
from chap_core.pandas_adaptors import get_time_period
from chap_core.rest_api_src.worker_functions import initialize_gee_client
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import PeriodRange, Month
from chap_core.util import interpolate_nans

csv_file_name = '~/Data/ch_data/dengue_data_ISIMIP.csv'
base_folder = '/home/knut/Data/ch_data/'
_polygon_filename = '%sgeometry/gadm41_VNM_1.json' % base_folder
polygon_base_filename = '%sgeometry/{country}.json' % base_folder
out_base_filename = '%sera5/{country}_era5.csv' % base_folder
full_data_basename = '%sfull_data/{country}.csv' % base_folder


class DFeatureModel(FeatureModel):
    properties: dict


class DFeatureCollectionModel(FeatureCollectionModel):
    features: list[DFeatureModel]


def to_camelcase(name: str) -> str:
    return ''.join([word.capitalize() for word in name.split()])


def normalize_name(name: str) -> str:
    return unidecode(to_camelcase(name))


def get_country_data(country_name, old_data=None):
    existing_names = [] if old_data is None else old_data.location.unique()
    polygon_filename = polygon_base_filename.format(country=country_name)

    polygons = DFeatureCollectionModel.model_validate_json(open(polygon_filename).read())
    polygon_dict = {unidecode(feature.properties['VARNAME_1']): feature for feature in polygons.features}
    polygon_dict2 = {unidecode(feature.properties['NAME_1']): feature for feature in polygons.features}
    print([feature.properties for feature in polygons.features])
    entries = df[df['country'] == country_name]

    periods = [pd.Period(f'{year}-{month}', 'M')
               for year, month in zip(entries['year'], entries['month'])]

    min_period = min(zip(entries['year'], entries['month']))
    max_period = max(zip(entries['year'], entries['month']))

    period_range = PeriodRange.from_time_periods(Month(*min_period), Month(*max_period))

    # myshpfile = geopandas.read_file(polygon_filename)
    # utf8_geom_names = set(myshpfile['VARNAME_1'].unique())

    gee_client = initialize_gee_client()
    features = []
    geo_names = set(polygon_dict.keys()) | set(polygon_dict2.keys())
    print(geo_names)
    print(entries['admin1'].unique())
    names = []
    for name in entries['admin1'].unique():
        if not isinstance(name, str):
            continue
        cname = to_camelcase(unidecode(name))
        if cname in existing_names:
            print(f'{cname} already in data')
            continue
        # print(cname, cname in polygon_dict)
        if cname in polygon_dict:
            features.append(polygon_dict[cname])
            names.append(cname)
        elif cname in polygon_dict2:
            features.append(polygon_dict2[cname])
            names.append(cname)
        else:
            print(f'Could not find {cname} in {geo_names}')
    if not features:
        print(f'Could not find any features for {country_name}')
        return None
    # names = [feature.properties['VARNAME_1'] for feature in features]
    data = gee_client.get_historical_era5(
        DFeatureCollectionModel(features=features).model_dump(),
        periodes=period_range)

    data = DataSet({name: value for name, value in zip(names, data.values())})
    return data


def get_climate_data():
    df = pd.read_csv(csv_file_name)
    countries = df['country'].unique()
    for country in countries:
        filename = out_base_filename.format(country=country)
        if os.path.exists(filename):
            old_data = pd.read_csv(filename)
        data = get_country_data(country, old_data)
        if data is not None:
            print(data)
            data.to_csv(filename, mode='a')


def join_climate_and_health_data():
    df: pd.DataFrame = pd.read_csv(csv_file_name)
    mask = df['admin1'].apply(lambda x: isinstance(x, str))
    df = df[mask]
    countries = df['country'].unique()

    for country in countries:
        print(f'Running for country: {country}')
        climate_data_set = out_base_filename.format(country=country)
        if not os.path.exists(climate_data_set):
            continue
        health_data = df[df['country'] == country]

        health_data['location'] = health_data['admin1'].apply(normalize_name)
        health_data = health_data.rename(columns={'dengue_cases': 'disease_cases',
                                                  'population_worldpop': 'population'})
        health_data['time_period'] = get_time_period(health_data, 'year', 'month')
        st_data = DataSet.from_pandas(health_data, dataclass=HealthPopulationData, fill_missing=True)
        climate_data = DataSet.from_csv(climate_data_set, dataclass=SimpleClimateData)
        # climate_data = climate_data.restrict_time_period(slice(st_data.period_range[0], st_data.period_range[-1]))
        print(st_data.period_range)
        print(climate_data.period_range)
        locations = set(st_data.locations()) & set(climate_data.locations())
        data = {location: FullData(time_period=st_data[location].time_period,
                                   disease_cases=st_data[location].disease_cases,
                                   population=interpolate_nans(st_data[location].population),
                                   mean_temperature=climate_data[location].mean_temperature,
                                   rainfall=climate_data[location].rainfall)
                for location in locations}
        st_data = DataSet(data)
        print(st_data)
        st_data.to_csv(full_data_basename.format(country=country))


def clean_files():
    for country in os.listdir(base_folder + 'era5'):
        filename = base_folder + 'era5/' + country
        lines = list(open(filename))
        with open(filename, 'w') as out_f:
            out_f.write(lines[0])
            for line in lines[1:]:
                first = line.strip().split(',')
                # check if int
                if first[0].isdigit():
                    out_f.write(line)


# get_climate_data()
if __name__ == '__main__':
    #clean_files()
    join_climate_and_health_data()
