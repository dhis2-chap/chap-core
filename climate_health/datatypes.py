import bionumpy as bnp


@bnp.bnpdataclass
class ClimateHealthTimeSeries:
    time_period: str
    rainfall: float
    mean_temperature: float
    disease_cases: int


@bnp.bnpdataclass
class LocatedClimateHealthTimeSeries(ClimateHealthTimeSeries):
    time_period: str
    rainfall: float
    mean_temperature: float
    location: str
    disease_cases: int

