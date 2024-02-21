from datetime import date
from .external import ee
import pandas as pd
import numpy as np
from ..datatypes import ClimateData, Location, TimePeriod, Shape
from ..time_period import Month, Day
from ..time_period.dataclasses import Month as MonthDataclass, Day as DayDataclass
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
        self._monthly_ic = get_image_collection('MONTHLY', 'ERA5')
        self._daily_ic = get_image_collection('DAILY', 'ERA5')

    def get_data(self, region: Shape, start_period: TimePeriod, end_period: TimePeriod, exclusive_end=True) -> ClimateData:
        assert isinstance(region, Location), f'Expected Location, got {type(region)}'
        assert isinstance(start_period, (Month, Day)), f'Expected Month, got {type(start_period)}, {start_period}'
        is_daily = hasattr(start_period, 'day')
        start_day = start_period.day if is_daily else 1
        start_date = f'{start_period.year}-{start_period.month:02d}-{start_day:02d}'
        end_data = self._get_end_date(end_period, exclusive_end)
        point = ee.Geometry.Point(region.longitude, region.latitude)
        variable_names = ['temperature_2m', 'temperature_2m_max', 'total_precipitation_sum']
        ic = self._daily_ic if is_daily else self._monthly_ic
        ic = ic.filterDate(start_date, end_data).select(*variable_names)
        data = ic.map(lambda image: ee.Feature(None, image.reduceRegion(ee.Reducer.mean(), point, 1)))
        info = data.getInfo()
        variable_dicts = {name: np.array([f['properties'][name] for f in info['features']]) for name in variable_names}
        ids = [v['id'] for v in info['features']]
        years, months = zip(*((int(id[:4]), int(id[4:6])) for id in ids))
        if not is_daily:
            periods = MonthDataclass(years, months)
        else:
            days = [int(id[6:8]) for id in ids]
            periods = DayDataclass(years, months, days)

        celcius_offset = 273.15
        return ClimateData(periods, rainfall=variable_dicts['total_precipitation_sum'], mean_temperature=variable_dicts['temperature_2m']-celcius_offset, max_temperature=variable_dicts['temperature_2m_max']-celcius_offset)

    def _get_end_date(self, end_period, exclusive_end):
        end_add = 0 if exclusive_end else 1
        if isinstance(end_period, Day):
            end_day = end_period.day + end_add
            end_month = end_period.month
            return f'{end_period.year}-{end_month:02d}-{end_day:02d}'
        elif isinstance(end_period, Month):
            end_data = f'{end_period.year}-{end_period.month + end_add:02d}-01'
        return end_data
