from chap_core.predictor.naive_estimator import NaiveEstimator
from chap_core.testing.estimators import sanity_check_estimator


def test_train():
    estimator = NaiveEstimator()
    sanity_check_estimator(estimator)
