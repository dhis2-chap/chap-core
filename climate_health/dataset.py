from typing import Protocol, Union, Iterable

from climate_health.datatypes import Location
from climate_health.time_period.dataclasses import Period

spatial_index_type = Union[str, Location]
temporal_index_type = Union[Period, Iterable[Period], slice]


class SpatioTemporalDataSet(Protocol):
    dataclass = ...
    def get_locations(self, location: Iterable[Location])-> SpatioTemporalDataSet:
        ...

    def get_data_for_location(self, location: Location)->dataclass
        ...
        
    def restrict_time_period(self, period_range: slice[Period]):
        ...
        
    # def __getitem__(self, index: spatial_index_type|Tuple[spatial_index_type, temporal_index_type] ) -> Union['SpatioTemporalDataSet', 'dataclass']:
    #     ...
