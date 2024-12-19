from typing import Optional, List

import bionumpy as bnp
import numpy as np
import pandas as pd
from bionumpy.bnpdataclass import BNPDataClass

from pydantic import BaseModel, field_validator
import dataclasses

from typing_extensions import deprecated

from .api_types import PeriodObservation
from .time_period import PeriodRange
from .time_period.date_util_wrapper import TimeStamp
from .util import interpolate_nans


def tsdataclass(cls):
    tmp_cls = bnp.bnpdataclass.bnpdataclass(cls)
    tmp_cls.__annotations__["time_period"] = PeriodRange
    return tmp_cls


@tsdataclass
class TimeSeriesData:
    time_period: PeriodRange

    def model_dump(self):
        return {field.name: getattr(self, field.name).tolist() for field in dataclasses.fields(self)}

    def __getstate__(self):
        return self.todict()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def join(self, other):
        return np.concatenate([self, other])

    def resample(self, freq):
        df = self.topandas()
        df["time_period"] = self.time_period.to_period_index()
        df = df.set_index('time_period')
        df = df.resample(freq).interpolate()
        return self.from_pandas(df.reset_index())

    def topandas(self):
        data_dict = {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray) and value.ndim > 1:
                data_dict[key] = value.tolist()
        data_dict["time_period"] = self.time_period.topandas()
        return pd.DataFrame(data_dict)

    to_pandas = topandas

    def to_csv(self, csv_file: str, **kwargs):
        """Write data to a csv file."""
        data = self.to_pandas()
        data.to_csv(csv_file, index=False, **kwargs)

    def to_pickle_dict(self):
        data_dict = {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}
        data_dict['time_period'] = self.time_period.tolist()
        return data_dict

    @classmethod
    def from_pickle_dict(cls, data: dict):
        return cls(
            **{key: PeriodRange.from_strings(value) if key == 'time_period' else value for key, value in data.items()})

    @classmethod
    def create_class_from_basemodel(cls, dataclass: type[PeriodObservation]):
        fields = dataclass.model_fields
        fields = [
            (name, field.annotation) if name != "time_period" else (name, PeriodRange) for name, field in fields.items()
        ]
        return dataclasses.make_dataclass(dataclass.__name__, fields, bases=(TimeSeriesData,))

    @staticmethod
    def _fill_missing(data, missing_indices):
        if len(missing_indices) == 0:
            return data
        n_entries = len(data) + len(missing_indices)
        filled_data = np.full(n_entries, np.nan)
        mask = np.full(n_entries, True)
        mask[missing_indices] = False
        filled_data[mask] = data
        return filled_data

    @classmethod
    def from_pandas(cls, data: pd.DataFrame, fill_missing=False) -> "TimeSeriesData":
        try:
            time_strings = data.time_period.astype(str)
            # check unique
            assert len(time_strings) == len(set(time_strings)), f'{time_strings} has duplicates'
            time = PeriodRange.from_strings(time_strings, fill_missing=fill_missing)
        except Exception:
            print("Error in time period: ", data.time_period)
            raise

        if fill_missing:
            time, missing_indices = time
            mask = np.full(len(time), True)
            mask[missing_indices] = False
        else:
            missing_indices = []
        # time = parse_periods_strings(data.time_period.astype(str))
        variable_names = [field.name for field in dataclasses.fields(cls) if field.name != "time_period"]
        data = [cls._fill_missing(data[name].values, missing_indices) for name in variable_names]
        assert all(len(d) == len(time) for d in data), f"{[len(d) for d in data]} != {len(time)}"
        return cls(time, **dict(zip(variable_names, data)))

    @classmethod
    def from_csv(cls, csv_file: str, **kwargs):
        """Read data from a csv file."""
        data = pd.read_csv(csv_file, **kwargs)
        return cls.from_pandas(data)

    def interpolate(self, field_names: Optional[List[str]] = None):
        data_dict = {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}
        data_dict["time_period"] = self.time_period
        fields = {
            key: interpolate_nans(value) if field_names is None or key in field_names else value
            for key, value in data_dict.items()
            if key != "time_period"
        }
        return self.__class__(self.time_period, **fields)

    @deprecated("Compatibility with old code")
    def data(self):
        return self

    @property
    def start_timestamp(self) -> pd.Timestamp:
        return self.time_period[0].start_timestamp

    @property
    def end_timestamp(self) -> pd.Timestamp:
        return self.time_period[-1].end_timestamp

    def fill_to_endpoint(self, end_time_stamp: TimeStamp) -> "TimeSeriesData":
        if self.end_timestamp == end_time_stamp:
            return self
        n_missing = (end_time_stamp - self.end_timestamp) // self.time_period.delta
        assert n_missing >= 0, (f"{n_missing} < 0", end_time_stamp, self.end_timestamp)
        old_time_period = self.time_period
        new_time_period = PeriodRange(old_time_period.start_timestamp, end_time_stamp, old_time_period.delta)
        d = {field.name: getattr(self, field.name) for field in dataclasses.fields(self) if field.name != "time_period"}

        for name, data in d.items():
            d[name] = np.pad(data.astype(float), (0, n_missing), constant_values=np.nan)
        return self.__class__(new_time_period, **d)

    def fill_to_range(self, start_timestamp, end_timestamp):
        if self.end_timestamp == end_timestamp and self.start_timestamp == start_timestamp:
            return self
        n_missing_start = self.time_period.delta.n_periods(start_timestamp, self.start_timestamp)
        # (self.start_timestamp - start_timestamp) // self.time_period.delta
        n_missing = self.time_period.delta.n_periods(self.end_timestamp, end_timestamp)
        # n_missing = (end_timestamp - self.end_timestamp) // self.time_period.delta
        assert n_missing >= 0, (f"{n_missing} < 0", end_timestamp, self.end_timestamp)
        assert n_missing_start >= 0, (
            f"{n_missing} < 0",
            end_timestamp,
            self.end_timestamp,
        )
        old_time_period = self.time_period
        new_time_period = PeriodRange(start_timestamp, end_timestamp, old_time_period.delta)
        d = {field.name: getattr(self, field.name) for field in dataclasses.fields(self) if field.name != "time_period"}

        for name, data in d.items():
            d[name] = np.pad(data.astype(float), (n_missing_start, n_missing), constant_values=np.nan)
        return self.__class__(new_time_period, **d)

    def to_array(self):
        return np.array(
            [getattr(self, field.name) for field in dataclasses.fields(self) if field.name != "time_period"]
        ).T

    def todict(self):
        d = super().todict()
        d["time_period"] = self.time_period.topandas()
        return d

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            **{key: PeriodRange.from_strings(value) if key == 'time_period' else value for key, value in data.items()})

    def merge(self, other: 'TimeSeriesData', result_class: type['TimeSeriesData']):
        data_dict = {}
        if len(self.time_period) != len(other.time_period) or np.any(self.time_period != other.time_period):
            raise ValueError(f"{self.time_period} != {other.time_period}")
        for field in dataclasses.fields(result_class):
            field_name = field.name
            if field_name == "time_period":
                continue
            if hasattr(self, field_name):
                assert not hasattr(other, field_name), f"Field {field_name} in both data"
                data_dict[field_name] = getattr(self, field_name)
            elif hasattr(other, field_name):
                data_dict[field_name] = getattr(other, field_name)
            else:
                raise ValueError(f"Field {field_name} not in either data")
        return result_class(self.time_period, **data_dict)


@tsdataclass
class TimeSeriesArray(TimeSeriesData):
    value: float


@tsdataclass
class SimpleClimateData(TimeSeriesData):
    rainfall: float
    mean_temperature: float
    # max_temperature: float


@tsdataclass
class ClimateData(TimeSeriesData):
    rainfall: float
    mean_temperature: float
    max_temperature: float


@tsdataclass
class HealthData(TimeSeriesData):
    disease_cases: int


@tsdataclass
class ClimateHealthTimeSeries(TimeSeriesData):
    rainfall: float
    mean_temperature: float
    disease_cases: int

    @classmethod
    def combine(
            cls, health_data: HealthData, climate_data: ClimateData, fill_missing=False
    ) -> "ClimateHealthTimeSeries":
        return ClimateHealthTimeSeries(
            time_period=health_data.time_period,
            rainfall=climate_data.rainfall,
            mean_temperature=climate_data.mean_temperature,
            disease_cases=health_data.disease_cases,
        )


ClimateHealthData = ClimateHealthTimeSeries


@tsdataclass
class FullData(ClimateHealthData):
    population: int

    @classmethod
    def combine(
            cls, health_data: HealthData, climate_data: ClimateData, population: float
    ) -> "ClimateHealthTimeSeries":
        return cls(
            time_period=health_data.time_period,
            rainfall=climate_data.rainfall,
            mean_temperature=climate_data.mean_temperature,
            disease_cases=health_data.disease_cases,
            population=np.full(len(health_data), population),
        )


@tsdataclass
class LocatedClimateHealthTimeSeries(ClimateHealthTimeSeries):
    location: str


class ClimateHealthTimeSeriesModel(BaseModel):
    time_period: str | pd.Period
    rainfall: float
    mean_temperature: float
    disease_cases: int

    class Config:
        arbitrary_types_allowed = True

    @field_validator("time_period")
    def parse_time_period(cls, data: str | pd.Period) -> pd.Period:
        if isinstance(data, pd.Period):
            return data
        else:
            return pd.Period(data)


@tsdataclass
class HealthPopulationData(HealthData):
    population: int


class Shape:
    pass


@dataclasses.dataclass
class Location(Shape):
    latitude: float
    longitude: float


@tsdataclass
class SummaryStatistics(TimeSeriesData):
    mean: float
    median: float
    std: float
    min: float
    max: float
    quantile_low: float
    quantile_high: float
    # quantile_size: -> Maybe add this later


@tsdataclass
class Samples(TimeSeriesData):
    samples: float

    def topandas(self):
        n_samples = self.samples.shape[-1]
        df = pd.DataFrame(
            {"time_period": self.time_period.topandas()} | {f"sample_{i}": self.samples[:, i] for i in range(n_samples)}
        )
        return df

    @classmethod
    def from_pandas(cls, data: pd.DataFrame, fill_missing=False) -> "TimeSeriesData":
        ptime = PeriodRange.from_strings(data.time_period.astype(str), fill_missing=fill_missing)
        n_samples = sum(1 for col in data.columns if col.startswith("sample_"))
        samples = np.array([data[f"sample_{i}"].values for i in range(n_samples)]).T
        return cls(ptime, samples)

    to_pandas = topandas

    def summaries(self, q_low=0.25, q_high=0.75):
        return SummaryStatistics(
            self.time_period,
            mean=np.mean(self.samples, axis=-1),
            median=np.median(self.samples, axis=-1),
            std=np.std(self.samples, axis=-1),
            min=np.min(self.samples, axis=-1),
            max=np.max(self.samples, axis=-1),
            quantile_low=np.quantile(self.samples, q_low, axis=-1),
            quantile_high=np.quantile(self.samples, q_high, axis=-1),
        )


@dataclasses.dataclass
class Quantile:
    low: float
    high: float
    size: float


ResultType = pd.DataFrame


def add_field(data: BNPDataClass, new_class: type, **field_data):
    return new_class(**{field.name: getattr(data, field.name) for field in dataclasses.fields(data)} | field_data)


def remove_field(data: BNPDataClass, field_name, new_class=None):
    is_type = isinstance(data, type)
    if new_class is None:
        new_class = tsdataclass(
            dataclasses.make_dataclass(
                data.__class__.__name__,
                [(field.name, field.type) for field in dataclasses.fields(data) if field.name != field_name],
                bases=(TimeSeriesData,),
            )
        )
        if is_type:
            return new_class
    return new_class(
        **{field.name: getattr(data, field.name) for field in dataclasses.fields(data) if field.name != field_name}
    )


@tsdataclass
class GEEData(TimeSeriesData):
    temperature_2m: float
    total_precipitation_sum: float


@tsdataclass
class FullGEEData(HealthPopulationData):
    temperature_2m: float
    total_precipitation_sum: float


def create_tsdataclass(field_names):
    return tsdataclass(
        dataclasses.make_dataclass(
            "TimeSeriesData",
            [(name, float) for name in field_names],
            bases=(TimeSeriesData,),
        )
    )
