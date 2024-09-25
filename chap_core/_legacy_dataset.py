from typing import Protocol, TypeAlias, Union, Iterable, TypeVar, Tuple

import pandas as pd
from pydantic import BaseModel

from chap_core.datatypes import Location
from chap_core.time_period.dataclasses import Period

SpatialIndexType: TypeAlias = Union[str, Location]
TemporalIndexType: TypeAlias = Union[Period, Iterable[Period], slice]


FeaturesT = TypeVar("FeaturesT")


class ClimateData(BaseModel):
    temperature: float
    rainfall: float
    humidity: float


class DataType(BaseModel):
    disease_cases: int
    climate_data: ClimateData


class IsTemporalDataSet(Protocol[FeaturesT]):
    def restrict_time_period(
        self, start_period: Period = None, end_period: Period = None
    ) -> "IsTemporalDataSet[FeaturesT]": ...

    def to_tidy_dataframe(self) -> pd.DataFrame: ...

    @classmethod
    def from_tidy_dataframe(cls, df: pd.DataFrame) -> "IsSpatialDataSet[FeaturesT]": ...


class TemporalArray:
    def __init__(self, time_index, data):
        self._time_index = time_index
        self._data = data

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return ufunc


class IsSpatialDataSet(Protocol[FeaturesT]):
    def get_locations(
        self, location: Iterable[SpatialIndexType]
    ) -> "IsSpatialDataSet[FeaturesT]": ...

    def get_location(self, location: SpatialIndexType) -> FeaturesT: ...

    def locations(self) -> Iterable[SpatialIndexType]: ...

    def data(self) -> Iterable[FeaturesT]: ...

    def location_items(self) -> Iterable[Tuple[SpatialIndexType, FeaturesT]]: ...

    def to_tidy_dataframe(self) -> pd.DataFrame: ...

    @classmethod
    def from_tidy_dataframe(cls, df: pd.DataFrame) -> "IsSpatialDataSet[FeaturesT]": ...


class IsSpatioTemporalDataSet(Protocol[FeaturesT]):
    dataclass = ...

    def get_data_for_locations(
        self, location: Iterable[SpatialIndexType]
    ) -> "IsSpatioTemporalDataSet[FeaturesT]": ...

    def get_data_for_location(self, location: SpatialIndexType) -> FeaturesT: ...

    def restrict_time_period(
        self, start_period: Period = None, end_period: Period = None
    ) -> "IsSpatioTemporalDataSet[FeaturesT]": ...

    def start_time(self) -> Period: ...

    def end_time(self) -> Period: ...

    def locations(self) -> Iterable[SpatialIndexType]: ...

    def data(self) -> Iterable[FeaturesT]: ...

    def location_items(
        self,
    ) -> Iterable[Tuple[SpatialIndexType, IsTemporalDataSet[FeaturesT]]]: ...

    def to_tidy_dataframe(self) -> pd.DataFrame: ...

    @classmethod
    def from_tidy_dataframe(
        cls, df: pd.DataFrame
    ) -> "IsSpatioTemporalDataSet[FeaturesT]": ...

    def to_csv(self, file_name: str): ...

    @classmethod
    def from_csv(self, file_name: str) -> "IsSpatioTemporalDataSet[FeaturesT]": ...
