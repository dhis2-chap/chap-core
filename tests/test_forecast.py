from chap_core.assessment.forecast import forecast_ahead
from chap_core.data.datasets import ISIMIP_dengue_harmonized
from chap_core.predictor.naive_estimator import NaiveEstimator
from chap_core.time_period.date_util_wrapper import delta_month


def test_forecast_ahead():
    dataset = ISIMIP_dengue_harmonized["vietnam"]
    forecast = forecast_ahead(NaiveEstimator(), dataset, 3)  # type: ignore[reportArgumentType]
    assert set(forecast.locations()) == set(dataset.locations())
    assert len(forecast.period_range) == 3
    assert forecast.period_range[0] == dataset.period_range[-1] + delta_month
