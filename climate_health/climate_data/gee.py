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
        point = ee.Geometry.Point(region.longitude, region.latitude)
        variable_names = ['temperature_2m', 'temperature_2m_max', 'total_precipitation_sum']
        ic = self._ic.filterDate(start_date, end_data).select(*variable_names)
        data = ic.map(lambda image: ee.Feature(None, image.reduceRegion(ee.Reducer.mean(), point, 1)))
        info = data.getInfo()
        variable_dicts = {name: np.array([f['properties'][name] for f in info['features']]) for name in variable_names}
        years, months = zip(*((int(v['id'][:4]), int(v['id'][5:7])) for v in info['features']))
        periods = MonthDataclass(years, months)
        return ClimateData(periods, rainfall=variable_dicts['total_precipitation_sum'], mean_temperature=variable_dicts['temperature_2m'], max_temperature=variable_dicts['temperature_2m_max'])
