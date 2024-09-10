from climate_health.assessment.dataset_splitting import train_test_split_with_weather
from climate_health.plotting.prediction_plot import plot_forecast_from_summaries
from climate_health.spatio_temporal_data.temporal_dataclass import DataSet
from climate_health.time_period.date_util_wrapper import TimeDelta, Month
import logging

logger = logging.getLogger(__name__)


def forecast(model, dataset: DataSet, prediction_length: TimeDelta, graph=None):
    '''
    Forecast n_months into the future using the model
    '''
    logger.info(f'Forecasting {prediction_length} months into the future')
    split_point = dataset.end_timestamp - prediction_length
    split_period = Month(split_point.year, split_point.month)
    train_data, test_set, future_weather = train_test_split_with_weather(dataset, split_period)
    if graph is not None and hasattr(model, 'set_graph'):
        model.set_graph(graph)

    model._num_warmup = 1000
    model._num_samples = 400
    model.train(train_data)
    if False and hasattr(model, 'diagnose'):
        model.diagnose()
    predictions = model.forecast(future_weather, 10, prediction_length)
    return predictions


def multi_forecast(model, dataset: DataSet, prediction_lenght: TimeDelta, pre_train_delta: TimeDelta):
    '''
    Forecast n_months into the future using the model
    '''
    cur_dataset = dataset
    datasets = []
    init_timestamp = dataset.start_timestamp + pre_train_delta + prediction_lenght
    while cur_dataset.end_timestamp > init_timestamp:
        datasets.append(cur_dataset)
        split_point = cur_dataset.end_timestamp - prediction_lenght
        split_period = Month(split_point.year, split_point.month)
        cur_dataset, _, _ = train_test_split_with_weather(cur_dataset, split_period)
    logger.info(f'Forecasting {prediction_lenght} months into the future on {len(datasets)} datasets')
    return (forecast(model, dataset, prediction_lenght) for dataset in datasets[::-1])
