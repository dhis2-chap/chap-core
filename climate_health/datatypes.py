import bionumpy as bnp
import pandas as pd


@bnp.bnpdataclass.bnpdataclass
class ClimateHealthTimeSeries:
    time_period: str
    rainfall: float
    mean_temperature: float
    disease_cases: int

    @classmethod
    def from_csv(cls, csv_file: str, **kwargs):
        """Read data from a csv file."""
        data = pd.read_csv(csv_file, dtype={'time_period': str}, **kwargs)
        return cls(data.time_period, data.rainfall, data.mean_temperature, data.disease_cases)

    def to_csv(self, csv_file: str, **kwargs):
        """Write data to a csv file."""
        data = pd.DataFrame({
            "time_period": self.time_period,
            "rainfall": self.rainfall,
            "mean_temperature": self.mean_temperature,
            "disease_cases": self.disease_cases,
        })
        data.to_csv(csv_file, index=False, **kwargs)


@bnp.bnpdataclass.bnpdataclass
class LocatedClimateHealthTimeSeries(ClimateHealthTimeSeries):
    time_period: str
    rainfall: float
    mean_temperature: float
    location: str
    disease_cases: int

