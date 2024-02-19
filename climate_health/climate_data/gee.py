from datetime import date
from .external import ee
import pandas as pd
import numpy as np
from ..datatypes import ClimateData, Location, TimePeriod, Shape
from ..time_period import Month
from ..time_period.dataclasses import Month as MonthDataclass
import numpy as np


def get_image_collection(period='MONTHLY', dataset='ERA5'):
    dataset_lookup = {'ERA5': 'ECMWF/ERA5_LAND'}
    name = f'{dataset_lookup[dataset]}/{period}_AGGR'
    ic = ee.ImageCollection(name)
    return ic

class ERA5DataBase:
    def __init__(self):
        #ee.Authenticate()
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        self._ic = get_image_collection('MONTHLY', 'ERA5')

    def get_data(self, region: Shape, start_period: TimePeriod, end_period: TimePeriod, exclusive_end=True) -> ClimateData:
        assert isinstance(region, Location), f'Expected Location, got {type(region)}'
        assert isinstance(start_period, Month)
        start_date = f'{start_period.year}-{start_period.month:02d}-01'
        end_add = 1 if exclusive_end else 0
        end_data = f'{end_period.year}-{end_period.month+exclusive_end:02d}-01'
        variable_names = ['temperature_2m', 'temperature_2m_max', 'total_precipitation_sum']
        ic = self._ic.filterDate(start_date, end_data).select(*variable_names)
        point = ee.Geometry.Point(region.longitude, region.latitude)
        data = ic.map(lambda image: ee.Feature(None, image.reduceRegion(ee.Reducer.mean(), point, 1)))
        info = data.getInfo()
        variable_dicts = {name: np.array([f['properties'][name] for f in info['features']]) for name in variable_names}
        years, months = zip(*((int(v['id'][:4]), int(v['id'][5:7])) for v in info['features']))
        periods = MonthDataclass(years, months)
        return ClimateData(periods, rainfall=variable_dicts['total_precipitation_sum'], mean_temperature=variable_dicts['temperature_2m'])


'''
class EEWrapper(SpatioTemporalIndexable):
    def __init__(self, ee_object, name=None):
        self._ee_object = ee_object
        self._name = name

    def _get_temporal_item(self, item: TimePoint):
        if isinstance(item, slice):
            value = self._ee_object.filterDate(item.start, item.stop)
            return self._wrap(value)
        else:
            raise ValueError(f'Cannot index {self.__class__.__name__} with {item}')

    def _get_spatial_item(self, item):
        # if isinstance(item, ee.Geometry.Point):
        if isinstance(item, list):
            #use image.RedcueRegions instead
            # FeatureCollection from a list of features.
            # list_of_features = [
            #     ee.Feature(ee.Geometry.Point(-62.54, -27.32), {'key': 'val1'}),
            #     ee.Feature(ee.Geometry.Point(-69.18, -10.64), {'key': 'val2'}),
            #     ee.Feature(ee.Geometry.Point(-45.98, -18.09), {'key': 'val3'})
            # ]
            #list_of_features_fc = ee.FeatureCollection(list_of_features)
            pass
        value = self._ee_object.map(lambda image: ee.Feature(None, image.reduceRegion(ee.Reducer.mean(), item, 1)))
        return self._wrap(value)

    def _wrap(self, value):
        return self.__class__(value, name=self._name)

    def __getattr__(self, name):
        # Also get two names
        return self.__class__(self._ee_object.select(name), name=name)

    def __repr__(self):
        return repr(self._ee_object)

    def __str__(self):
        return str(self._ee_object)

    def compute(self):
        assert self._name is not None
        features = self._ee_object.getInfo()['features']
        return np.array([f['properties'][self._name] for f in features])
'''

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November',
          'December']

lookup = dict(zip(months, range(12)))

def get_city_name(location):
    return location.split(maxsplit=6)[-1]


def get_date(month, year):
    return date(year, lookup[month]+1, 1)

def get_dataset(filename):
    filepath_or_buffer = '../example_data/10yeardengudata.csv'
    data = pd.read_csv(filename, sep='\t', header=1)
    years = np.array([int(s.strip().split()[-1]) for s in data['periodname']])
    data['periodname'] = [get_date(s.strip().split()[0], int(s.strip().split()[-1])) for s in data['periodname']]
    # month = [lookup[s.strip().split()[0]] for s in data['periodname']]
    # data['periodname'] = (years * 12 + month)
    data = data.sort_values(by='periodname')
    data = data.iloc[:-2] # remove november, december 2023
    for columnname in data.columns[1:]:
        column = data[columnname]
        data[columnname] = column.replace(np.nan, 0)

    return data


def main():
    data = get_dataset('../example_data/10yeardengudata.csv')
    print(data)

    # data = pd.melt(data, id_vars='periodname', var_name='location')
    # data['year'] = [str(year) for year in data['periodname'] // 12]
    # data['month'] = data['periodname'] % 12
    # print(data)
    # print(min(data['periodname']), max(data['periodname']))
    # city_names = [get_city_name(c) for c in data.columns[1:]]
    # locations = [get_location(name) for name in city_names]
    # point_dict = {name: location for name, location in zip(data.columns, locations)}
    # pickle point dict
    import pickle
    # with open('point_dict.pickle', 'wb') as f:
    #    pickle.dump(point_dict, f)

    # read point_dict
    with open('point_dict.pickle', 'rb') as f:
        point_dict = pickle.load(f)

    # point_dict = eval(open('point_dict.py').read())
    print(point_dict)
    import os
    name = list(point_dict)[1]
    if os.path.exists(f'{name}.csv'):
        new_data_frame = pd.read_csv(f'{name}.csv')
        analyze_data(new_data_frame, exog_names=['Rainfall'])
        return

    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    ic = get_image_collection()
    ee_dataset = EEWrapper(ic)
    range_data = extract_same_range_from_climate_data(data, ee_dataset)
    d = range_data.total_precipitation_sum
    d2 = range_data.temperature_2m
    d3 = range_data.temperature_2m_max
    dataset[start_date:stop_data:time.week]

    for name, point in list(point_dict.items())[1:2]:
        if point is None:
            continue
        ee_point = ee.Geometry.Point(point.longitude, point.latitude)
        print(name, ee_point)
        values = d[ee_point].compute()
        temperature = d2[ee_point].compute()
        new_data_frame = pd.DataFrame({'Date': data['periodname'],
                                       'Rainfall': values,
                                       'DengueCases': data[name],
                                       'Temperature': temperature})
        # data['precipitation'] = values
        print(name)
        new_data_frame.to_csv(f'{name}.csv')
        analyze_data(new_data_frame, exog_names=['Rainfall'])



    # with open('point_dict.py', 'w') as f:
    #     f.write(repr(point_dict))
    # for name, location in point_dict.items():
    #     print(location)
    #     print(data[name])





if __name__ == '__main__':
    main()