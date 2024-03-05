from typing import Protocol

from climate_health.dataset import SpatioTemporalDataSet
from climate_health.datatypes import ClimateHealthTimeSeries, ClimateData, HealthData


class ExternalModel(Protocol):
    def get_predictions(self, train_data: SpatioTemporalDataSet[ClimateHealthTimeSeries],
                        future_climate_data: SpatioTemporalDataSet[ClimateData]) -> SpatioTemporalDataSet[HealthData]:
        ...


