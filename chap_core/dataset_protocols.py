from datetime import datetime
from typing import Protocol, TypeVar, Generic, Iterable, Union, Tuple

import pandas as pd

from chap_core.datatypes import Location

TemporalIndex = Union[datetime, pd.Timestamp, slice]
SpatialIndex = Union[str, Location]
T = TypeVar("T")


class TemporalData(Generic[T]):
    def __getitem__(self, item: TemporalIndex) -> "TemporalData": ...

    def get_values(self) -> Iterable[T]: ...

    @property
    def start_time(self) -> datetime: ...

    @property
    def end_time(self) -> datetime: ...


class FixedResolutionTemporalData(TemporalData):
    def get_values(self) -> Iterable[T]: ...

    @property
    def start_time(self) -> datetime: ...

    @property
    def end_time(self) -> datetime: ...

    def topandas(self) -> pd.DataFrame: ...


class SpatialData(Generic[T]):
    def __getitem__(self, item: SpatialIndex) -> Union["SpatialData[T]", T]: ...


class SpatioTemporalData(Generic[T]):
    def __getitem__(
        self, item: Tuple[TemporalIndex, SpatialIndex]
    ) -> Union["SpatioTemporalData[T]", T]: ...


class IsSpatioTemporalDataSet(Protocol[T]):
    dataclass = ...

    def get_data_for_locations(
        self, location: Iterable[spatial_index_type]
    ) -> "IsSpatioTemporalDataSet[T]": ...

    def get_data_for_location(self, location: spatial_index_type) -> T: ...

    def restrict_time_period(
        self, start_period: Period = None, end_period: Period = None
    ) -> "IsSpatioTemporalDataSet[T]": ...

    def start_time(self) -> Period: ...

    def end_time(self) -> Period: ...

    def locations(self) -> Iterable[spatial_index_type]: ...

    def data(self) -> Iterable[T]: ...

    def items(self) -> Iterable[Tuple[spatial_index_type, T]]: ...

    def to_tidy_dataframe(self) -> pd.DataFrame: ...

    @classmethod
    def from_tidy_dataframe(cls, df: pd.DataFrame) -> "IsSpatioTemporalDataSet[T]": ...
