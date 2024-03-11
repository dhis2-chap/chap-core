from typing import Generic, Iterable, Tuple

import numpy as np
from bionumpy.bnpdataclass import BNPDataClass

from ..dataset import temporal_index_type, Features
from ..datatypes import Location


class TemporalDataclass:
    def __init__(self, data: BNPDataClass):
        self._data = data

    def restrict_time_period(self, period_range: temporal_index_type) -> 'TemporalDataclass':
        assert isinstance(period_range, slice)
        assert period_range.step is None
        mask = np.full_like(self._data.time_period, True)
        if period_range.start is not None:
            mask = mask & (self._data.time_period >= period_range.start)
        if period_range.stop is not None:
            mask = mask & (self._data.time_period < period_range.stop)
        return TemporalDataclass(self._data[mask])

    def data(self):
        return self._data


class SpatioTemporalDict(Generic[Features]):
    def __init__(self, data_dict: dict[Features]):
        self._data_dict = data_dict

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
