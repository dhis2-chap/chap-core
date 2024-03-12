from typing import Generic, Iterable, Tuple

import numpy as np
import pandas as pd

from ..dataset import temporal_index_type, Features
from ..datatypes import Location


class TemporalDataclass(Generic[Features]):
    '''
    Wraps a dataclass in a object that is can be sliced by time period.
    Call .data() to get the data back.
    '''

    def __init__(self, data: Features):
        self._data = data

    def __repr__(self):
        return f'{self.__class__.__name__}({self._data})'

    def restrict_time_period(self, period_range: temporal_index_type) -> 'TemporalDataclass[Features]':
        assert isinstance(period_range, slice)
        assert period_range.step is None
        mask = np.full(len(self._data.time_period), True)
        if period_range.start is not None:
            mask = mask & (self._data.time_period >= period_range.start)
        if period_range.stop is not None:
            mask = mask & (self._data.time_period <= period_range.stop)
        return TemporalDataclass(self._data[mask])

    def data(self) -> Iterable[Features]:
        return self._data

    def to_pandas(self):
        return self._data.to_pandas()


class SpatioTemporalDict(Generic[Features]):
    def __init__(self, data_dict: dict[Features]):
        self._data_dict = {loc: TemporalDataclass(data) if not isinstance(data, TemporalDataclass) else data for loc, data in data_dict.items()}

    def __repr__(self):
        return f'{self.__class__.__name__}({self._data_dict})'

    def get_locations(self, location: Iterable[Location]) -> 'SpatioTemporalDict[Features]':
        return self.__class__({loc: self._data_dict[loc] for loc in location})

    def get_location(self, location: Location) -> Features:
        return self._data_dict[location]

    def restrict_time_period(self, period_range: temporal_index_type) -> 'SpatioTemporalDict[Features]':
        return self.__class__(
            {loc: data.restrict_time_period(period_range) for loc, data in self._data_dict.items()})

    def locations(self) -> Iterable[Location]:
        return self._data_dict.keys()

    def data(self) -> Iterable[Features]:
        return self._data_dict.values()

    def items(self) -> Iterable[Tuple[Location, Features]]:
        return self._data_dict.items()

    def _add_location_to_dataframe(self, df, location):
        df['location'] = location
        return df

    def to_pandas(self):
        ''' Join the pandas frame for all locations with locations as column'''
        tables = [self._add_location_to_dataframe(data.to_pandas(), location) for location, data in self._data_dict.items()]
        return pd.concat(tables)