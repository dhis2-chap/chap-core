from dataclasses import dataclass, field
from typing import List, Dict


# Disease cases
@dataclass
class DiseaseObservation:
    time_period: str
    disease_cases: int


@dataclass
class DiseaseTimeSeries:
    observations: List[DiseaseObservation]


@dataclass
class MultiLocationDiseaseTimeSeries:
    timeseries_dict: Dict[str, DiseaseTimeSeries] = field(default_factory=dict)

    def __setitem__(self, location, timeseries):
        self.timeseries_dict[location] = timeseries

    def __getitem__(self, location):
        return self.timeseries_dict[location]

    def locations(self):
        return iter(self.timeseries_dict.keys())

    def timeseries(self):
        return iter(self.timeseries_dict.values())


# Assessment metric
@dataclass
class Error:
    time_period: str
    value: float


@dataclass
class ErrorTimeSeries:
    observations: List[Error]


@dataclass
class MultiLocationErrorTimeSeries:
    timeseries_dict: Dict[str, ErrorTimeSeries]

    def __getitem__(self, location):
        return self.timeseries_dict[location]

    def __setitem__(self, location, timeseries):
        self.timeseries_dict[location] = timeseries

    def locations(self):
        return iter(self.timeseries_dict.keys())

    def timeseries(self):
        return iter(self.timeseries_dict.values())

    def num_locations(self):
        return len(self.timeseries_dict)

    def num_timeperiods(self):
        return len(self.get_all_timeperiods())

    def get_the_only_location(self):
        assert len(self.timeseries_dict) == 1
        return list(self.timeseries_dict.keys())[0]

    def get_the_only_timeseries(self):
        assert len(self.timeseries_dict) == 1
        return list(self.timeseries_dict.values())[0]

    def get_all_timeperiods(self):
        timeperiods = None
        for ts in self.timeseries():
            current_timepriod_value = [o.time_period for o in ts.observations]
            if timeperiods is None:
                timeperiods = current_timepriod_value
            else:
                assert timeperiods == current_timepriod_value
        return timeperiods

    def timeseries_length(self):
        lengths = [len(ts.observations) for ts in self.timeseries()]
        assert len(set(lengths)) == 1
        return lengths[0]

    def locationvalues_per_timepoint(self) -> List[Dict[str, Error]]:
        return [
            dict([(location, timeseries.observations[i]) for location, timeseries in self.timeseries_dict.items()])
            for i in range(self.timeseries_length())
        ]


# Forecasts
@dataclass
class Samples:
    time_period: str
    disease_case_samples: List[float]


@dataclass
class Forecast:
    predictions: List[Samples]


@dataclass
class MultiLocationForecast:
    timeseries: Dict[str, Forecast]
