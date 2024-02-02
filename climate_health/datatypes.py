import bionumpy as bnp
import pandas as pd
from pydantic import BaseModel


@bnp.bnpdataclass.bnpdataclass
class ClimateHealthTimeSeries:
    time_period: str
    rainfall: float
    mean_temperature: float
    disease_cases: int

    @classmethod
    def from_csv(cls, csv_file: str, **kwargs):
        """Read data from a csv file."""
        data = pd.read_csv(csv_file, dtype={'Time': str}, **kwargs)
        return cls(data.Time, data.Rain, data.Temperature, data.Disease)

    def to_csv(self, csv_file: str, **kwargs):
        """Write data to a csv file."""
        data = pd.DataFrame({
            "Time": self.time_period,
            "Rain": self.rainfall,
            "Temperature": self.mean_temperature,
            "Disease": self.disease_cases,
        })
        data.to_csv(csv_file, index=False, **kwargs)


@bnp.bnpdataclass.bnpdataclass
class LocatedClimateHealthTimeSeries(ClimateHealthTimeSeries):
    time_period: str
    rainfall: float
    mean_temperature: float
    location: str
    disease_cases: int


class ClimateHealthTimeSeriesModel(BaseModel):
    time_period: str
    rainfall: float
    mean_temperature: float
    disease_cases: int


class LocatedClimateHealthTimeSeriesModel(BaseModel):
    time_period: str
    rainfall: float
    mean_temperature: float
    location: str
    disease_cases: int
