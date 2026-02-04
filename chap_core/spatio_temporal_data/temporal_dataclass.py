import dataclasses
import logging
import pickle
from numbers import Number
from pathlib import Path, PurePath
from typing import IO, Callable, Generic, Iterable, Optional, Protocol, Tuple, Type, TypeVar, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pydantic import BaseModel

from ..api_types import FeatureCollectionModel, PeriodObservation
from ..datatypes import (
    TimeSeriesArray,
    TimeSeriesData,
    add_field,
    create_tsdataclass,
    remove_field,
)
from ..geometry import Polygons
from ..time_period import Month, PeriodRange
from ..time_period.date_util_wrapper import TimeStamp, Week, clean_timestring

logger = logging.getLogger(__name__)


class TimeSeriesLike(Protocol):
    """Protocol for time series data types."""

    time_period: PeriodRange

    def to_pandas(self) -> pd.DataFrame: ...
    def to_pickle_dict(self) -> dict: ...
    def interpolate(self, field_names: Optional[list[str]] = None) -> "TimeSeriesLike": ...
    def fill_to_range(self, start_timestamp: TimeStamp, end_timestamp: TimeStamp) -> "TimeSeriesLike": ...
    def join(self, other: "TimeSeriesLike") -> "TimeSeriesLike": ...

    @classmethod
    def from_pandas(cls, data: pd.DataFrame, fill_missing: bool = False) -> "TimeSeriesLike": ...
    @classmethod
    def from_pickle_dict(cls, data: dict) -> "TimeSeriesLike": ...

    @property
    def start_timestamp(self) -> TimeStamp: ...
    @property
    def end_timestamp(self) -> TimeStamp: ...


FeaturesT = TypeVar("FeaturesT")
TemporalIndexType = slice


class TemporalDataclass(Generic[FeaturesT]):
    """
    Wraps a dataclass in a object that is can be sliced by time period.
    Call .data() to get the data back.
    """

    def __init__(self, data: FeaturesT):
        self._data = data

    def __repr__(self):
        return f"{self.__class__.__name__}({self._data})"

    def _restrict_by_slice(self, period_range: slice):
        assert period_range.step is None
        start, stop = (None, None)
        if period_range.start is not None:
            start = self._data.time_period.searchsorted(period_range.start)
        if period_range.stop is not None:
            stop = self._data.time_period.searchsorted(period_range.stop, side="right")
        return self._data[start:stop]  # type: ignore[index]

    def fill_to_endpoint(self, end_time_stamp: TimeStamp) -> "TemporalDataclass[FeaturesT]":
        if self.end_timestamp == end_time_stamp:
            return self
        n_missing = self._data.time_period.delta.n_periods(self.end_timestamp, end_time_stamp)
        # n_missing = (end_time_stamp - self.end_timestamp) // self._data.time_period.delta
        assert n_missing >= 0, (f"{n_missing} < 0", end_time_stamp, self.end_timestamp)
        old_time_period = self._data.time_period
        new_time_period = PeriodRange(old_time_period.start_timestamp, end_time_stamp, old_time_period.delta)
        d = {
            field.name: getattr(self._data, field.name)
            for field in dataclasses.fields(self._data)  # type: ignore[arg-type]
            if field.name != "time_period"
        }

        for name, data in d.items():
            d[name] = np.pad(data.astype(float), (0, n_missing), constant_values=np.nan)
        return self._data.__class__(new_time_period, **d)  # type: ignore[call-arg, return-value]

    def fill_to_range(self, start_timestamp, end_timestamp):
        if self.end_timestamp == end_timestamp and self.start_timestamp == start_timestamp:
            return self
        n_missing_start = self._data.time_period.delta.n_periods(start_timestamp, self.start_timestamp)
        n_missing = self._data.time_period.delta.n_periods(self.end_timestamp, end_timestamp)
        assert n_missing >= 0, (f"{n_missing} < 0", end_timestamp, self.end_timestamp)
        assert n_missing_start >= 0, (
            f"{n_missing} < 0",
            end_timestamp,
            self.end_timestamp,
        )
        old_time_period = self._data.time_period
        new_time_period = PeriodRange(start_timestamp, end_timestamp, old_time_period.delta)
        d = {
            field.name: getattr(self._data, field.name)
            for field in dataclasses.fields(self._data)  # type: ignore[arg-type]
            if field.name != "time_period"
        }

        for name, data in d.items():
            d[name] = np.pad(data.astype(float), (n_missing_start, n_missing), constant_values=np.nan)
        return self._data.__class__(new_time_period, **d)  # type: ignore[call-arg]

    def restrict_time_period(self, period_range: TemporalIndexType) -> "TemporalDataclass[FeaturesT]":
        assert isinstance(period_range, slice)
        assert period_range.step is None
        if hasattr(self._data.time_period, "searchsorted"):
            return self._restrict_by_slice(period_range)  # type: ignore[return-value, no-any-return]

        mask = np.full(len(self._data.time_period), True)

        if period_range.start is not None:
            mask = mask & (self._data.time_period >= period_range.start)

        if period_range.stop is not None:
            mask = mask & (self._data.time_period <= period_range.stop)

        return self._data[mask]  # type: ignore[return-value, index, no-any-return]

    def data(self) -> FeaturesT:
        return self._data

    def to_pandas(self) -> pd.DataFrame:
        return self._data.to_pandas()

    def join(self, other):
        return np.concatenate([self._data, other._data])

    @property
    def start_timestamp(self) -> TimeStamp:
        return self._data.time_period[0].start_timestamp  # type: ignore[no-any-return]

    @property
    def end_timestamp(self) -> TimeStamp:
        return self._data.time_period[-1].end_timestamp  # type: ignore[no-any-return]


class Polygon:
    pass


class DataSetMetaData(BaseModel):
    name: str = "dataset"
    filename: str | None = None
    db_id: int | None = None


class DataSet(Generic[FeaturesT]):
    """
    Class representing severeal time series at different locations.
    """

    def __init__(self, data_dict: dict[str, FeaturesT], polygons=None, metadata=DataSetMetaData()):
        self._data_dict = {loc: data for loc, data in data_dict.items()}
        self._polygons = polygons
        self._parent_dict = None
        self.metadata = metadata

    def field_names(self):
        return [
            field.name
            for field in dataclasses.fields(next(iter(self._data_dict.values())))  # type: ignore[arg-type]
            if field.name not in ("time_period", "location")
        ]

    @property
    def frequency(self):
        first_period = self.period_range[0]
        if isinstance(first_period, Month):
            return "M"
        elif isinstance(first_period, Week):
            return "W"
        else:
            raise NotImplementedError

    def model_dump(self):
        return {
            "data_dict": {loc: data.model_dump() for loc, data in self._data_dict.items()},  # type: ignore[attr-defined]
            "polygons": self._polygons and self._polygons.model_dump(),
        }

    @classmethod
    def from_dict(cls, data: dict, dataclass: type):  # type: ignore[type-arg]
        data_dict = {loc: dataclass.from_dict(val) for loc, val in data["data_dict"].items()}  # type: ignore[attr-defined]
        return cls(data_dict, data["polygons"] and FeatureCollectionModel(**data["polygons"]))

    def set_polygons(self, polygons: FeatureCollectionModel, ignore_validation: bool = False) -> list[str]:
        polygon_ids = {feature.id for feature in polygons.features}
        ignored_locations: set[str]

        if not ignore_validation:
            ignored_locations = set(self._data_dict.keys()) - polygon_ids
            if ignored_locations:
                logger.warning(
                    f"Found {len(ignored_locations)} locations in dataset that are not in the polygons: {ignored_locations}"
                )
            self._data_dict = {location: data for location, data in self._data_dict.items() if location in polygon_ids}
        else:
            ignored_locations = set()
            # for location in self.locations():
            #     if location not in polygon_ids:
            #         logger.warning(f"Found a location {location} (type: {type(location)}) in dataset ({location}) that is not in the polygons. Polygons contains: {polygon_ids}.  ")
            #         del self._data_dict[location]
            #    assert location in polygon_ids, f"Found a location {location} (type: {type(location)}) in dataset ({location}) that is not in the polygons. Polygons contains: {polygon_ids}.  "
        self._polygons = polygons
        return list(ignored_locations)

    def get_parent_dict(self) -> Optional[dict[str, str]]:
        if not self._polygons:
            return {str(location): "-" for location in self.locations()}
        return Polygons(self._polygons).get_parent_dict()  # type: ignore[return-value, no-any-return]

    def aggregate_to_parent(self, field_name: str = "disease_cases", nan_indicator: str = "disease_cases"):
        parent_dict = self.get_parent_dict()
        assert parent_dict is not None
        dataclass = create_tsdataclass([field_name])
        new_dict = {}
        period_range = self.period_range
        for location, data in self.items():
            parent = parent_dict[location]

            new_data = getattr(data, field_name).copy()
            if parent not in new_dict:
                new_dict[parent] = dataclass(period_range, np.zeros_like(new_data))
            old_data = getattr(new_dict[parent], field_name)
            new_data = getattr(data, field_name).copy()
            if nan_indicator is not None:
                nan_mask = np.isnan(getattr(data, nan_indicator))
                new_data[nan_mask] = 0
            old_data += new_data

        return self.__class__(new_dict, self._polygons)

    @property
    def polygons(self):
        return self._polygons

    def __repr__(self):
        return f"{self.__class__.__name__}({self._data_dict})"

    def __getitem__(self, location: str) -> FeaturesT:
        return self._data_dict[location]

    def keys(self) -> Iterable[str]:
        return self._data_dict.keys()

    def items(self) -> Iterable[Tuple[str, FeaturesT]]:
        return ((k, d) for k, d in self._data_dict.items())

    def values(self) -> Iterable[FeaturesT]:
        return (d for d in self._data_dict.values())

    @property
    def period_range(self) -> PeriodRange:
        try:
            first_period_range = self._data_dict[next(iter(self._data_dict))].time_period
        except StopIteration:
            raise ValueError(f"No data in dataset {self}")

        assert first_period_range.start_timestamp == first_period_range.start_timestamp
        assert first_period_range.end_timestamp == first_period_range.end_timestamp
        return first_period_range

    @property
    def start_timestamp(self) -> TimeStamp:
        return min(data.start_timestamp for data in self.data())

    @property
    def end_timestamp(self) -> TimeStamp:
        return max(data.end_timestamp for data in self.data())

    def get_locations(self, location: Iterable[str]) -> "DataSet[FeaturesT]":
        return self.__class__({loc: self._data_dict[loc] for loc in location}, self._polygons)

    def get_location(self, location: str) -> FeaturesT:
        return self._data_dict[location]

    def restrict_time_period(self, period_range: TemporalIndexType) -> "DataSet[FeaturesT]":
        return self.__class__(
            {loc: TemporalDataclass(data).restrict_time_period(period_range) for loc, data in self._data_dict.items()},  # type: ignore[misc]
            self._polygons,
        )

    def filter_locations(self, locations: Iterable[str]) -> "DataSet[FeaturesT]":
        return self.__class__({loc: data for loc, data in self.items() if loc in locations})

    def locations(self) -> Iterable[str]:
        return self._data_dict.keys()

    def data(self) -> Iterable[FeaturesT]:
        return self._data_dict.values()

    def _add_location_to_dataframe(self, df, location, field_name="location"):
        df[field_name] = location
        return df

    def _add_location_info_to_dataframe(self, df, location, parent_dict):
        if parent_dict is not None:
            df["parent"] = parent_dict[location]
        df["location"] = location
        return df

    def to_pandas(self) -> pd.DataFrame:
        """Join the pandas frame for all locations with locations as column"""
        parent_dict = self.get_parent_dict()

        try:
            tables = [
                self._add_location_info_to_dataframe(data.to_pandas(), location, parent_dict)
                for location, data in self._data_dict.items()
            ]
        except KeyError:
            logger.error(f"KeyError while looking up {self._data_dict.keys()} in {parent_dict}")
            raise
        return pd.concat(tables)  # type: ignore[return-value, no-any-return]

    def interpolate(self, field_names: Optional[list[str]] = None):
        return self.__class__({loc: data.interpolate(field_names) for loc, data in self.items()}, self._polygons)  # type: ignore[misc]

    @classmethod
    def _fill_missing(cls, data_dict: dict[str, FeaturesT]) -> dict[str, FeaturesT]:
        """Fill missing values in a dictionary of FeaturesT"""
        end = max(data.end_timestamp for data in data_dict.values())
        start = min(data.start_timestamp for data in data_dict.values())
        for location, data in data_dict.items():
            data_dict[location] = data.fill_to_range(start, end)  # type: ignore[assignment]
        return data_dict

    @classmethod
    def from_pandas(
        cls, df: pd.DataFrame, dataclass: Type[FeaturesT] | None = None, fill_missing: bool = False
    ) -> "DataSet[FeaturesT]":
        """
        Create a SpatioTemporalDict from a pandas dataframe.
        The dataframe needs to have a 'location' column, and a 'time_period' column.
        The time_period columnt needs to have strings that can be parsed into a period.
        All fields in the dataclass needs to be present in the dataframe.
        If 'fill_missing' is True, missing values will be filled with np.nan. Else all the time series needs to be
        consecutive.


        Parameters
        ----------
        df : pd.DataFrame
            The dataframe
        dataclass : Type[FeaturesT]
            The dataclass to use for the time series
        fill_missing : bool, optional
            If missing values should be filled, by default False

        Returns
        -------
        DataSet[FeaturesT]
            The SpatioTemporalDict

        Examples
        --------
        >>> import pandas as pd
        >>> from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
        >>> from chap_core.datatypes import HealthData
        >>> df = pd.DataFrame(
        ...     {
        ...         "location": ["Oslo", "Oslo", "Bergen", "Bergen"],
        ...         "time_period": ["2020-01", "2020-02", "2020-01", "2020-02"],
        ...         "disease_cases": [10, 20, 30, 40],
        ...     }
        ... )
        >>> DataSet.from_pandas(df, HealthData)
        """
        if dataclass is None:
            dataclass = create_tsdataclass(
                [col for col in df.columns.tolist() if col not in ("location", "time_period") and "Unnamed" not in col]
            )

        data_dict = {}
        non_string_locations = []
        for location, data in df.groupby("location"):
            if not isinstance(location, str):
                non_string_locations.append(location)
                location = str(location)

            time_element = data["time_period"].iloc[0]
            if isinstance(time_element, str) or isinstance(time_element, Number):
                # if time periods are string, clean them and convert to periods
                data["time_period"] = data["time_period"].apply(clean_timestring)

            data_dict[location] = dataclass.from_pandas(data.sort_values(by="time_period"), fill_missing)  # type: ignore[union-attr]
        data_dict = cls._fill_missing(data_dict)  # type: ignore[arg-type, assignment]

        if non_string_locations:
            logging.warning(f"{len(non_string_locations)} location(s) are not strings, converting to strings")

        return cls(data_dict)  # type: ignore[arg-type]

    def to_csv(self, file_name: str, mode="w"):
        self.to_pandas().to_csv(file_name, mode=mode)

    def to_pickle(self, file_name: str):
        data_dict = {loc: data.to_pickle_dict() for loc, data in self.items()}
        with open(file_name, "wb") as f:
            pickle.dump(data_dict, f)

    @classmethod
    def from_pickle(cls, file_name: str, dataclass: Type[FeaturesT]) -> "DataSet[FeaturesT]":
        with open(file_name, "rb") as f:
            data_dict = pickle.load(f)
        return cls({loc: dataclass.from_pickle_dict(val) for loc, val in data_dict.items()})  # type: ignore[misc]

    @classmethod
    def from_file(cls, file_name: str, dataclass: Type[FeaturesT]) -> "DataSet[FeaturesT]":
        if file_name.endswith(".csv"):
            return cls.from_csv(file_name, dataclass)
        if file_name.endswith(".pkl"):
            return cls.from_pickle(file_name, dataclass)
        raise ValueError("Unknown file type")

    @classmethod
    def df_from_pydantic_observations(cls, observations: list[PeriodObservation]) -> TimeSeriesData:
        df = pd.DataFrame([obs.model_dump() for obs in observations])
        dataclass = TimeSeriesData.create_class_from_basemodel(type(observations[0]))
        return dataclass.from_pandas(df)  # type: ignore[return-value, no-any-return]

    @classmethod
    def from_period_observations(
        cls, observation_dict: dict[str, list[PeriodObservation]]
    ) -> "DataSet[TimeSeriesData]":
        """
        Create a SpatioTemporalDict from a dictionary of PeriodObservations.
        The keys are the location names, and the values are lists of PeriodObservations.

        Parameters
        ----------
        observation_dict : dict[str, list[PeriodObservation]]
            The dictionary of observations

        Returns
        -------
        DataSet[TimeSeriesData]
            The SpatioTemporalDict

        Examples
        --------
        >>> from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
        >>> from chap_core.api_types import PeriodObservation
        >>> class HealthObservation(PeriodObservation):
        ...     disease_cases: int
        >>> observations = {
        ...     "Oslo": [
        ...         HealthObservation(time_period="2020-01", disease_cases=10),
        ...         HealthObservation(time_period="2020-02", disease_cases=20),
        ...     ]
        ... }
        >>> DataSet.from_period_observations(observations)
        >>> DataSet.to_pandas()
        """
        data_dict = {}
        for location, observations in observation_dict.items():
            data_dict[location] = cls.df_from_pydantic_observations(observations)
        return cls(data_dict)  # type: ignore[return-value, arg-type]

    @classmethod
    def from_csv(
        cls, file_name: Union[str, Path, IO[bytes]], dataclass: Type[FeaturesT] | None = None
    ) -> "DataSet[FeaturesT]":
        csv = pd.read_csv(file_name)
        if dataclass is None:
            dataclass = create_tsdataclass(
                [col for col in csv.columns.tolist() if col not in ("location", "time_period") and "Unnamed" not in col]
            )
        obj = cls.from_pandas(csv, dataclass)

        if isinstance(file_name, (str, Path)):
            path = Path(file_name).with_suffix(".geojson")
            if path.exists():
                with open(path, "r") as f:
                    obj.set_polygons(FeatureCollectionModel.model_validate_json(f.read()))
            else:
                path = Path(file_name).with_suffix(".json")
                if path.exists():
                    polygons = Polygons.from_file(path, id_property="NAME_1")
                    with open(path, "r") as f:
                        obj.set_polygons(polygons.feature_collection())
        if isinstance(file_name, (str, PurePath)):
            meta_data = DataSetMetaData(name=str(Path(file_name).stem), filename=str(file_name))
            obj.metadata = meta_data
        return obj

    def join_on_time(self, other: "DataSet[FeaturesT]") -> "DataSet[FeaturesT]":
        """Join two SpatioTemporalDicts on time. Returns a new SpatioTemporalDict.
        Assumes other is later in time.
        """
        return self.__class__(
            {loc: self._data_dict[loc].join(other._data_dict[loc]) for loc in self.locations()},  # type: ignore[misc]
            self._polygons,
        )

    def add_fields(self, new_type: type, **kwargs: Callable):
        return self.__class__(
            {
                loc: add_field(
                    data,
                    new_type,
                    **{key: func(data) for key, func in kwargs.items()},
                )
                for loc, data in self.items()
            }
        )

    def remove_field(self, field_name, new_class=None):
        return self.__class__(
            {loc: remove_field(data, field_name, new_class) for loc, data in self.items()}, self._polygons
        )

    @classmethod
    def from_fields(
        cls,
        dataclass: type[TimeSeriesData],
        fields: dict[str, "DataSet[TimeSeriesArray]"],
    ):
        start_timestamp = min(data.start_timestamp for data in fields.values())
        end_timestamp = max(data.end_timestamp for data in fields.values())
        period_range = PeriodRange(
            start_timestamp,
            end_timestamp,
            fields[next(iter(fields))].period_range.delta,
        )
        new_dict = {}
        field_names = list(fields.keys())
        # all_locations = {location for field in fields.values() for location in field.keys()}
        common_locations = set.intersection(*[set(field.keys()) for field in fields.values()])
        # for field, data in fields.items():
        #    assert set(data.keys()) == all_locations, (field, all_locations-set(data.keys()))
        for location in common_locations:
            new_dict[location] = dataclass(  # type: ignore[call-arg]
                period_range,
                **{
                    field: fields[field][location].fill_to_range(start_timestamp, end_timestamp).value  # type: ignore[union-attr]
                    for field in field_names
                },
            )
        return cls(new_dict)  # type: ignore[arg-type]

    def merge(self, other_dataset: "DataSet", result_dataclass: type[TimeSeriesData]) -> "DataSet":
        polygons_in_merged = None
        if self.polygons is not None and other_dataset.polygons is not None:
            raise Exception("Trying to merge two datasets with polygons, not sure how to do this (not implemented yet)")
        elif self.polygons is not None:
            polygons_in_merged = self.polygons
        elif other_dataset.polygons is not None:
            polygons_in_merged = self.polygons
        other_locations = set(other_dataset.locations())
        assert all(location in other_locations for location in self.locations()), (self.locations(), other_locations)
        new_dataset: DataSet = DataSet(
            {location: self[location].merge(other_dataset[location], result_dataclass) for location in self.locations()}  # type: ignore[misc, attr-defined]
        )
        if polygons_in_merged is not None:
            new_dataset.set_polygons(polygons_in_merged)
        return new_dataset

    def plot(self):
        for location, value in self.items():
            df = value.to_pandas()
            df.plot(x="time_period", y="disease_cases")
            plt.title(location)
        return plt

    def plot_aggregate(self):
        import plotly.express as px

        total = np.zeros(len(self.period_range))
        for location, value in self.items():
            total += np.where(np.isnan(getattr(value, "disease_cases")), 0, getattr(value, "disease_cases"))
        return px.line(x=self.period_range.tolist(), y=total)

    def to_report(self, pdf_filename: str):
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(pdf_filename) as pdf:
            for location, value in self.items():
                df = value.to_pandas()
                df.plot(x="time_period", y="disease_cases")
                df.plot(x="time_period", y="population")
                plt.title(location)
                pdf.savefig()
                plt.close()

    def resample(self, freq: str):
        return self.__class__({loc: data.resample(freq) for loc, data in self.items()}, self._polygons)  # type: ignore[attr-defined, misc]
