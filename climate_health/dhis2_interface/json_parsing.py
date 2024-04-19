import numpy as np
import pandas as pd

from climate_health.datatypes import HealthData, HealthPopulationData
from climate_health.dhis2_interface.src.PushResult import DataValue
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict


class MetadDataLookup:
    def __init__(self, meta_data_json):
        self._lookup = {name: value['name'] for name, value in meta_data_json['items'].items()}

    def __getitem__(self, item):
        return self._lookup[item]

    def __contains__(self, item):
        return item in self._lookup


def _get_period_id(time_period):
    if 'W' in time_period:
        year, week = time_period.split('W')
        return int(year) * 53 + int(week)
    else:
        year = time_period[:4]
        month = time_period[4:]
        return int(year) * 12 + int(month)


def parse_population_data(json_data, field_name='GEN - Population'):
    meta_data = MetadDataLookup(json_data['metaData'])
    lookup = {}
    for row in json_data['rows']:
        #if meta_data[row[0]] != field_name:
        #    continue
        lookup[row[2]] = int(row[3])
    return lookup


def _convert_time_period_string(row):
    if len(row) == 6 and 'W' not in row:
        return f'{row[:4]}-{row[4:]}'
    return row

def parse_disease_data(json_data, disease_name='IDS - Dengue Fever (Suspected cases)',
                       name_mapping={'time_period': 1, 'disease_cases': 3, 'location': 2}):
    meta_data = MetadDataLookup(json_data['metaData'])
    new_rows = []
    col_names = ['time_period', 'disease_cases', 'location']
    for row in json_data['rows']:
        # if meta_data[row[0]] != disease_name:
        #    continue
        new_row = row
        # new_row[name_mapping['location']] = meta_data[new_row[name_mapping['location']]]
        # new_row = [meta_data[elem] if elem in meta_data else elem for elem in row]
        new_rows.append([new_row[name_mapping[col_name]] for col_name in col_names])

    df = pd.DataFrame(new_rows, columns=col_names)
    df['week_id'] = [_get_period_id(row) for row in df['time_period']]
    df['time_period'] = [_convert_time_period_string(row) for row in df['time_period']]
    df.sort_values(by=['location', 'week_id'], inplace=True)
    return SpatioTemporalDict.from_pandas(df, dataclass=HealthData, fill_missing=True)


def join_data(json_data, population_data):
    population_lookup = parse_population_data(population_data)
    disease_data = parse_disease_data(json_data)
    return add_population_data(disease_data, population_lookup)


def add_population_data(disease_data, population_lookup):
    new_dict = {location: HealthPopulationData(data.data().time_period,
                                               data.data().disease_cases,
                                               np.full(len(data.data()), population_lookup[location])
                                               )
                for location, data in disease_data.items()}
    return SpatioTemporalDict(new_dict)


def predictions_to_json(data: SpatioTemporalDict[HealthData], attribute_mapping: dict[str, str]):
    entries = []
    for location, data in data.items():
        data = data.data()
        for i, time_period in enumerate(data.time_period):
            for from_name, to_name in attribute_mapping.items():

                entry = {'orgUnit': location,
                         'period': time_period.to_string(),
                         'value': getattr(data, from_name)[i],
                         'dataElement': to_name}
                entry = DataValue(getattr(data, from_name)[i],
                                  location,
                                  to_name,
                                  time_period.to_string().replace('-', ''))

                entries.append(entry)
    return entries
