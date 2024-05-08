from climate_health.assessment.forecast import forecast
from climate_health.file_io.example_data_set import datasets
from climate_health.plotting.prediction_plot import plot_forecast_from_summaries
from climate_health.predictor import get_model
from climate_health.time_period.date_util_wrapper import delta_month


def test_forecast():
    model = get_model('ewars_Plus')()
    dataset = datasets['hydromet_10'].load()
    predictions = forecast(model, dataset, 12*delta_month)
    for location, prediction in predictions.items():
        fig = plot_forecast_from_summaries(prediction.data(), dataset.get_location(location).data())
        fig.show()
