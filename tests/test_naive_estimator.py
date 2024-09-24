from climate_health.predictor.naive_estimator import NaiveEstimator
from climate_health.testing.estimators import sanity_check_estimator


def test_train():
    estimator = NaiveEstimator()
    sanity_check_estimator(estimator)
