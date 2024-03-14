from typing import Generic, Iterable, Tuple, Type

import numpy as np
import pandas as pd

from ..dataset import TemporalIndexType, FeaturesT
from ..datatypes import Location


class TemporalDataclass(Generic[FeaturesT]):
    '''
    Wraps a dataclass in a object that is can be sliced by time period.
    Call .data() to get the data back.
    '''

    def __init__(self, data: FeaturesT):
        self._data = data

    def __repr__(self):
        return f'{self.__class__.__name__}({self._data})'

    def _restrict_by_slice(self, period_range: slice):
        assert period_range.step is None
        mask = np.full(len(self._data.time_period), True)
        start, stop = (None, None)
        if period_range.start is not None:
            start = self._data.time_period.searchsorted(period_range.start)
        if period_range.stop is not None:
            stop = self._data.time_period.searchsorted(period_range.stop)
        return self._data[start:stop]

    def restrict_time_period(self, period_range: TemporalIndexType) -> 'TemporalDataclass[FeaturesT]':
        assert isinstance(period_range, slice)
        assert period_range.step is None
        if hasattr(self._data.time_period, 'searchsorted'):
            return TemporalDataclass(self._restrict_by_slice(period_range))
        mask = np.full(len(self._data.time_period), True)
        if period_range.start is not None:
            mask = mask & (self._data.time_period >= period_range.start)
        if period_range.stop is not None:
            mask = mask & (self._data.time_period <= period_range.stop)
        return TemporalDataclass(self._data[mask])

    def data(self) -> Iterable[FeaturesT]:
        return self._data

    def to_pandas(self) -> pd.DataFrame:
        return self._data.to_pandas()


class SpatioTemporalDict(Generic[FeaturesT]):
    def __init__(self, data_dict: dict[FeaturesT]):
        self._data_dict = {loc: TemporalDataclass(data) if not isinstance(data, TemporalDataclass) else data for loc, data in data_dict.items()}

    def __repr__(self):
        return f'{self.__class__.__name__}({self._data_dict})'

    def get_locations(self, location: Iterable[Location]) -> 'SpatioTemporalDict[FeaturesT]':
        return self.__class__({loc: self._data_dict[loc] for loc in location})

    def get_location(self, location: Location) -> FeaturesT:
        return self._data_dict[location]

    def restrict_time_period(self, period_range: TemporalIndexType) -> 'SpatioTemporalDict[FeaturesT]':
        return self.__class__(
            {loc: data.restrict_time_period(period_range) for loc, data in self._data_dict.items()})

    def locations(self) -> Iterable[Location]:
        return self._data_dict.keys()

    def data(self) -> Iterable[FeaturesT]:
        return self._data_dict.values()

    def items(self) -> Iterable[Tuple[Location, FeaturesT]]:
        return self._data_dict.items()

    def _add_location_to_dataframe(self, df, location):
        df['location'] = location
        return df

    def to_pandas(self) -> pd.DataFrame:
        ''' Join the pandas frame for all locations with locations as column'''
        tables = [self._add_location_to_dataframe(data.to_pandas(), location) for location, data in self._data_dict.items()]
        return pd.concat(tables)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame, dataclass: Type[FeaturesT]) -> 'SpatioTemporalDict[FeaturesT]':
        ''' Split a pandas frame into a SpatioTemporalDict'''
        data_dict = {}
        for location, data in df.groupby('location'):
            data_dict[location] = TemporalDataclass(dataclass.from_pandas(data))
        return cls(data_dict)

    def to_csv(self, file_name: str):
        self.to_pandas().to_csv(file_name)

    @classmethod
    def from_csv(cls, file_name: str, dataclass: Type[FeaturesT]) -> 'SpatioTemporalDict[FeaturesT]':
        return cls.from_pandas(pd.read_csv(file_name), dataclass)

