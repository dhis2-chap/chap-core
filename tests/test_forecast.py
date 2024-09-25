import pytest

from chap_core.assessment.forecast import forecast, multi_forecast, forecast_ahead
from chap_core.data.datasets import ISIMIP_dengue_harmonized
from chap_core.file_io.example_data_set import datasets
from chap_core.plotting.prediction_plot import plot_forecast_from_summaries
from chap_core.predictor import get_model
from chap_core.predictor.naive_estimator import NaiveEstimator
from chap_core.time_period.date_util_wrapper import delta_month


# @pytest.mark.skip(reason="Needs docked image")
@pytest.mark.skip(reason="slow")
def test_forecast():
    model = get_model("HierarchicalStateModelD2")(num_warmup=20, num_samples=20)
    dataset = datasets["hydromet_5_filtered"].load()
    predictions = forecast(model, dataset, 12 * delta_month)
    for location, prediction in predictions.items():
        fig = plot_forecast_from_summaries(
            prediction.data(), dataset.get_location(location).data()
        )
        fig.show()


@pytest.mark.skip(reason="slow")
def test_multi_forecast():
    model = get_model("HierarchicalStateModelD2")(num_warmup=20, num_samples=20)
    dataset = datasets["hydromet_5_filtered"].load()
    predictions_list = list(
        multi_forecast(
            model, dataset, 48 * delta_month, pre_train_delta=24 * delta_month
        )
    )
    for location, true_data in dataset.items():
        local_predictions = [
            pred.get_location(location).data() for pred in predictions_list
        ]
        fig = plot_forecast_from_summaries(local_predictions, true_data.data())
        fig.show()


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
