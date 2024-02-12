import bionumpy as bnp
import pandas as pd
from pydantic import BaseModel

from .file_io import parse_periods_strings
from .time_period.dataclasses import Period


@bnp.bnpdataclass.bnpdataclass
class ClimateHealthTimeSeries:
    time_period: Period
    rainfall: float
    mean_temperature: float
    disease_cases: int

    def todict(self):
        d = super().todict()
        d['time_period'] = self.time_period.topandas()
        return d

    @classmethod
    def from_csv(cls, csv_file: str, **kwargs):
        """Read data from a csv file."""
        data = pd.read_csv(csv_file, dtype={'Time': str}, **kwargs)
        return cls.from_pandas(data)

    @classmethod
    def from_pandas(cls, data):
        time = parse_periods_strings(data.Time)
        return cls(time, data.Rain, data.Temperature, data.Disease)

    def topandas(self):
        data = pd.DataFrame({
            "Time": self.time_period.topandas(),
            "Rain": self.rainfall,
            "Temperature": self.mean_temperature,
            "Disease": self.disease_cases,
        })
        return data

    def to_csv(self, csv_file: str, **kwargs):
        """Write data to a csv file."""
        data = self.to_pandas()
        data.to_csv(csv_file, index=False, **kwargs)


@bnp.bnpdataclass.bnpdataclass
class LocatedClimateHealthTimeSeries(ClimateHealthTimeSeries):
    location: str


class ClimateHealthTimeSeriesModel(BaseModel):
    time_period: Period
    rainfall: float
    mean_temperature: float
    disease_cases: int


class LocatedClimateHealthTimeSeriesModel(BaseModel):
    time_period: Period
    rainfall: float
    mean_temperature: float
    location: str
    disease_cases: int
