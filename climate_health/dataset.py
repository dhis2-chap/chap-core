from typing import Protocol, Union, Iterable

from climate_health.datatypes import Location
from climate_health.time_period.dataclasses import Period

spatial_index_type = Union[str, Location]
temporal_index_type = Union[Period, Iterable[Period], slice]


class SpatioTemporalDataSet(Protocol):
    dataclass = ...
    def __getitem__(self, index: int) -> Union['SpatioTemporalDataSet', 'dataclass']:
        ...
