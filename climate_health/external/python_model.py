from ..datatypes import ClimateHealthTimeSeries, HealthData, ClimateData
from ..dataset import SpatioTemporalDataSet

from climate_health.time_period import Month


class ExternalPythonModel:
    def __init__(self, script: str, lead_time=Month, adaptors=None):
        self.script = script
        self.lead_time = lead_time
        self.adaptors = adaptors

    def get_predictions(self, train_data: SpatioTemporalDataSet[ClimateHealthTimeSeries],
                        future_climate_data: SpatioTemporalDataSet[ClimateData]) -> SpatioTemporalDataSet[HealthData]:
        pass


