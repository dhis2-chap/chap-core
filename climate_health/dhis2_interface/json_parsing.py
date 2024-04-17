import pandas as pd

from climate_health.datatypes import HealthData
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict


class MetadDataLookup:
    def __init__(self, meta_data_json):
        self._lookup = {name: value['name'] for name, value in meta_data_json['items'].items()}

    def __getitem__(self, item):
        return self._lookup[item]

    def __contains__(self, item):
        return item in self._lookup


def _get_week_id(time_period):
    year, week = time_period.split('W')
    return int(year) * 53 + int(week)


def parse_json(json_data, disease_name='IDS - Dengue Fever (Suspected cases)',
               name_mapping={'time_period': 1, 'disease_cases': 3, 'location': 2}):
    meta_data = MetadDataLookup(json_data['metaData'])
    new_rows = []
    col_names = ['time_period', 'disease_cases', 'location']
    for row in json_data['rows']:
        if meta_data[row[0]] != disease_name:
            continue
        new_row = row
        # new_row[name_mapping['location']] = meta_data[new_row[name_mapping['location']]]
        # new_row = [meta_data[elem] if elem in meta_data else elem for elem in row]
        new_rows.append([new_row[name_mapping[col_name]] for col_name in col_names])

    df = pd.DataFrame(new_rows, columns=col_names)
    df['week_id'] = [_get_week_id(row) for row in df['time_period']]
    df.sort_values(by=['location', 'week_id'], inplace=True)
    return SpatioTemporalDict.from_pandas(df, dataclass=HealthData, fill_missing=True)
