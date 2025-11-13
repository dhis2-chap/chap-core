import pytest

from chap_core.assessment.forecast import forecast, multi_forecast, forecast_ahead
from chap_core.data.datasets import ISIMIP_dengue_harmonized
from chap_core.file_io.example_data_set import datasets
from chap_core.plotting.prediction_plot import plot_forecast_from_summaries
from chap_core.predictor import get_model
from chap_core.predictor.naive_estimator import NaiveEstimator
from chap_core.time_period.date_util_wrapper import delta_month


@pytest.fixture(scope="session")
def hydromet_dataset():
    dataset = datasets["hydromet_5_filtered"].load()
    return dataset


def test_forecast_ahead():
    model = NaiveEstimator()
    dataset = ISIMIP_dengue_harmonized["vietnam"]
    prediction_length = 3
    forecast_ahead(model, dataset, prediction_length)


def test_forecast_with_predicted_weather():
    model = NaiveEstimator()
    dataset = ISIMIP_dengue_harmonized["vietnam"]
    prediction_length = 3
    forecast_ahead(model, dataset, prediction_length)
