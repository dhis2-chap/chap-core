import logging

from chap_core.assessment.dataset_splitting import train_test_split_with_weather
from chap_core.assessment.prediction_evaluator import Estimator, Predictor
from chap_core.assessment.weather_providers import DEFAULT_WEATHER_PROVIDER_ID, get_future_weather
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period.date_util_wrapper import Month, PeriodRange, TimeDelta
from chap_core.validators import validate_training_data

logger = logging.getLogger(__name__)


def forecast(model, dataset: DataSet, prediction_length: TimeDelta, graph=None):
    """
    Forecast n_months into the future using the model
    """
    logger.info(f"Forecasting {prediction_length} months into the future")
    split_point = dataset.end_timestamp - prediction_length
    split_period = Month(split_point.year, split_point.month)
    train_data, _test_set, future_weather = train_test_split_with_weather(dataset, split_period)
    if graph is not None and hasattr(model, "set_graph"):
        model.set_graph(graph)

    model._num_warmup = 1000
    model._num_samples = 400
    model.train(train_data)
    if False:
        model.diagnose()
    predictions = model.forecast(future_weather, 10, prediction_length)
    return predictions


def multi_forecast(model, dataset: DataSet, prediction_lenght: TimeDelta, pre_train_delta: TimeDelta):
    """
    Forecast n_months into the future using the model
    """
    cur_dataset = dataset
    datasets = []
    init_timestamp = dataset.start_timestamp + pre_train_delta + prediction_lenght
    while cur_dataset.end_timestamp > init_timestamp:
        datasets.append(cur_dataset)
        split_point = cur_dataset.end_timestamp - prediction_lenght
        split_period = Month(split_point.year, split_point.month)
        cur_dataset, _, _ = train_test_split_with_weather(cur_dataset, split_period)
    logger.info(f"Forecasting {prediction_lenght} months into the future on {len(datasets)} datasets")
    return (forecast(model, dataset, prediction_lenght) for dataset in datasets[::-1])


def forecast_ahead(
    estimator: Estimator,
    dataset: DataSet,
    prediction_length: int,
    weather_provider: str = DEFAULT_WEATHER_PROVIDER_ID,
):
    """
    Forecast n_months into the future using the model
    """
    logger.info(f"Forecasting {prediction_length} months into the future")
    train_data = dataset
    validate_training_data(train_data, estimator)
    predictor = estimator.train(train_data)
    return forecast_with_predicted_weather(
        predictor,
        train_data,
        prediction_length,
        weather_provider=weather_provider,
    )


def forecast_with_predicted_weather(
    predictor: Predictor,
    historic_data: DataSet,
    prediction_length: int,
    weather_provider: str = DEFAULT_WEATHER_PROVIDER_ID,
):
    delta = historic_data.period_range[0].time_delta
    prediction_range = PeriodRange(
        historic_data.end_timestamp,
        historic_data.end_timestamp + delta * prediction_length,
        delta,
    )
    # No future_data exists when predicting ahead for real, so a provider that
    # declares leaks_future_data will correctly refuse here.
    future_weather = get_future_weather(weather_provider, historic_data, prediction_range)
    predictions = predictor.predict(historic_data, future_weather)
    return predictions
