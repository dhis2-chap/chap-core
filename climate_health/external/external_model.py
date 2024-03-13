from typing import Protocol

from climate_health.dataset import IsSpatioTemporalDataSet
from climate_health.datatypes import ClimateHealthTimeSeries, ClimateData, HealthData


class IsExternalModel(Protocol):
    def get_predictions(self, train_data: IsSpatioTemporalDataSet[ClimateHealthTimeSeries],
                        future_climate_data: IsSpatioTemporalDataSet[ClimateData]) -> IsSpatioTemporalDataSet[HealthData]:
        ...


