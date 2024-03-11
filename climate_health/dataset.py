from typing import Protocol, Union, Iterable, TypeVar, Generic, Tuple

import numpy as np
import pandas as pd
from prefect.utilities import pydantic
from pydantic import BaseModel

from climate_health.datatypes import Location, ClimateData
from climate_health.time_period import dataclasses
from climate_health.time_period.dataclasses import Period

spatial_index_type = Union[str, Location]
temporal_index_type = Union[Period, Iterable[Period], slice]

Features = TypeVar('Features')


class ClimateData(BaseModel):
    temperature: float
    rainfall: float
    humidity: float


class DataType(BaseModel):
    disease_cases: int
    climate_data: ClimateData



class TemporalDataSet(Protocol, Generic[Features]):
    def restrict_time_period(self, start_period: Period=None, end_period: Period=None) -> 'TemporalDataSet[Features]':
        ...

    def to_tidy_dataframe(self) -> pd.DataFrame:
        ...

    @classmethod
    def from_tidy_dataframe(cls, df: pd.DataFrame) -> 'SpatialDataSet[Features]':
        ...

class TemporalArray:
    def __init__(self, time_index, data):
        self._time_index = time_index
        self._data = data

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return ufunc9

class SpatialDataSet(Protocol, Generic[Features]):
    def get_locations(self, location: Iterable[spatial_index_type]) -> 'SpatialDataSet[Features]':
        ...

    def get_location(self, location: spatial_index_type) -> Features:
        ...

    def locations(self) -> Iterable[spatial_index_type]:
        ...

    def data(self) -> Iterable[Features]:
        ...

    def location_items(self) -> Iterable[Tuple[spatial_index_type, Features]]:
        ...

    def to_tidy_dataframe(self) -> pd.DataFrame:
        ...

    @classmethod
    def from_tidy_dataframe(cls, df: pd.DataFrame) -> 'SpatialDataSet[Features]':
        ...

class SpatioTemporalDataSet(Protocol, Generic[Features]):
    dataclass = ...

    def get_data_for_locations(self, location: Iterable[spatial_index_type]) -> 'SpatioTemporalDataSet[Features]':
        ...

    def get_data_for_location(self, location: spatial_index_type) -> Features:
        ...

    def restrict_time_period(self, start_period: Period=None, end_period: Period=None) -> 'SpatioTemporalDataSet[Features]':
        ...

    def start_time(self) -> Period:
        ...

    def end_time(self) -> Period:
        ...

    def locations(self) -> Iterable[spatial_index_type]:
        ...

    def data(self) -> Iterable[Features]:
        ...

    def location_items(self) -> Iterable[Tuple[spatial_index_type, TemporalDataSet[Features]]]:
        ...

    def to_tidy_dataframe(self) -> pd.DataFrame:
        ...

    @classmethod
    def from_tidy_dataframe(cls, df: pd.DataFrame) -> 'SpatioTemporalDataSet[Features]':
        ...

K = TypeVar('K')

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
        return NotImplemented

    def locations(self) -> Iterable[Location]:
        return self._data_dict.keys()

    def data(self) -> Iterable[Features]:
        return self._data_dict.values()

    def items(self) -> Iterable[Tuple[Location, Features]]:
        return self._data_dict.items()