import pytest

from climate_health.assessment.forecast import forecast, multi_forecast
from climate_health.file_io.example_data_set import datasets
from climate_health.plotting.prediction_plot import plot_forecast_from_summaries
from climate_health.predictor import get_model
from climate_health.time_period.date_util_wrapper import delta_month


# @pytest.mark.skip(reason="Needs docked image")
def test_forecast():
    model = get_model('HierarchicalStateModelD2')(num_warmup=20, num_samples=20)
    dataset = datasets['hydromet_5_filtered'].load()
    predictions = forecast(model, dataset, 12*delta_month)
    for location, prediction in predictions.items():
        fig = plot_forecast_from_summaries(prediction.data(), dataset.get_location(location).data())
        fig.show()


def test_multi_forecast():
    model = get_model('HierarchicalStateModelD2')(num_warmup=20, num_samples=20)
    dataset = datasets['hydromet_5_filtered'].load()
    predictions_list = list(multi_forecast(model, dataset, 48*delta_month, pre_train_delta=24*delta_month))
    for location, true_data in dataset.items():
        local_predictions = [pred.get_location(location).data() for pred in predictions_list]
        fig = plot_forecast_from_summaries(local_predictions, true_data.data())
        fig.show()
