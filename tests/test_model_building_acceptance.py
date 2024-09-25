import pytest

from climate_health._legacy_dataset import IsSpatioTemporalDataSet
from climate_health.datatypes import HealthData
from climate_health.predictor.naive_predictor import NaiveForecastSampler
from climate_health.time_period import Day
from .mocks import ClimateDataBaseMock
# from omnipy import MultiModelDataset


@pytest.mark.xfail
def test_model_building_acceptance(
    health_data_set_filename: str, output_file_name: str
):
    health_data = HealthData.from_csv(health_data_set_filename)
    data_set: IsSpatioTemporalDataSet = add_climate_data_to_health_data(
        health_data, ClimateDataBaseMock(), resolution=Day
    )
    model = NaiveForecastSampler
    results: MultiModelDataset = evaluate_model_on_multiple_time_series(model, data_set)
    present_results(results)
