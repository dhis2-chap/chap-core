import bionumpy as bnp


@bnp.bnpdataclass
class ClimateHealthTimeSeries:
    time_period: str
    rainfall: float
    mean_temperature: float


@bnp.bnpdataclass
class LocatedClimateHealthTimeSeries(ClimateHealthTimeSeries):
    time_period: str
    rainfall: float
    mean_temperature: float
    location: str

