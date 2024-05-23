from climate_health.assessment.dataset_splitting import train_test_split_with_weather
from climate_health.plotting.prediction_plot import plot_forecast_from_summaries
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
from climate_health.time_period.date_util_wrapper import TimeDelta, Month


def forecast(model, dataset: SpatioTemporalDict, prediction_length: TimeDelta):
    '''
    Forecast n_months into the future using the model
    '''
    split_point = dataset.end_timestamp - prediction_length
    split_period = Month(split_point.year, split_point.month)
    train_data, test_set, future_weather = train_test_split_with_weather(dataset, split_period)
    model._num_warmup = 1000
    model._num_samples = 400
    model.train(train_data)
    if hasattr(model, 'diagnose'):
        ... #model.diagnose()
    predictions = model.forecast(future_weather, 10, prediction_length)
    return predictions
