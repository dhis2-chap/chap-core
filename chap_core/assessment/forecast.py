import logging

from chap_core.assessment.prediction_evaluator import Estimator, Predictor
from chap_core.assessment.weather_providers import DEFAULT_WEATHER_PROVIDER_ID, get_future_weather
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period.date_util_wrapper import PeriodRange
from chap_core.validators import validate_training_data

logger = logging.getLogger(__name__)


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
