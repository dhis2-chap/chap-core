from typing import Protocol, Union, Iterable, TypeVar, Generic, Tuple

from climate_health.datatypes import Location
from climate_health.time_period.dataclasses import Period

spatial_index_type = Union[str, Location]
temporal_index_type = Union[Period, Iterable[Period], slice]

T = TypeVar('T')


class SpatioTemporalDataSet(Protocol, Generic[T]):
    dataclass = ...

    def get_data_for_locations(self, location: Iterable[spatial_index_type]) -> 'SpatioTemporalDataSet[T]':
        ...

    def get_data_for_location(self, location: spatial_index_type) -> T:
        ...

    def restrict_time_period(self, period_range: temporal_index_type) -> 'SpatioTemporalDataSet[T]':
        ...

    def locations(self) -> Iterable[spatial_index_type]:
        ...

    def data(self) -> Iterable[T]:
        ...

    def items(self) -> Iterable[Tuple[spatial_index_type, T]]:
        ...


class SpatioTemporalDict(Generic[T]):
    def __init__(self, data_dict: dict[Location, T]):
        self._data_dict = data_dict

    def get_locations(self, location: Iterable[Location]) -> 'SpatioTemporalDict[T]':
        return self.__class__({loc: self._data_dict[loc] for loc in location})

    def get_location(self, location: Location) -> T:
        return self._data_dict[location]

    def restrict_time_period(self, period_range: temporal_index_type) -> 'SpatioTemporalDict[T]':
        return NotImplemented

    def locations(self) -> Iterable[Location]:
        return self._data_dict.keys()

    def data(self) -> Iterable[T]:
        return self._data_dict.values()

    def items(self) -> Iterable[Tuple[Location, T]]:
        return self._data_dict.items()