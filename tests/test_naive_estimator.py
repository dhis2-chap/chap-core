from unittest.mock import patch

from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB, ModelTemplateDB
from chap_core.predictor.naive_estimator import NaiveEstimator
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.testing.estimators import sanity_check_estimator


def test_train():
    estimator = NaiveEstimator()
    sanity_check_estimator(estimator)


def test_predict_without_csv_round_trip(train_data, future_climate_data):
    predictor = NaiveEstimator().train(train_data)

    # Predictions must not depend on a CSV file shared between concurrent jobs.
    with patch.object(DataSet, "to_csv", side_effect=AssertionError("Prediction must not write a CSV")):
        samples = predictor.predict(train_data, future_climate_data, num_samples=7)

    assert set(samples.locations()) == set(future_climate_data.locations())
    for location, expected in future_climate_data.items():
        assert list(samples[location].time_period) == list(expected.time_period)
        assert samples[location].samples.shape == (len(expected), 7)


def test_model_metadata_class_attributes():
    assert isinstance(NaiveEstimator.model_template_db, ModelTemplateDB)
    assert isinstance(NaiveEstimator.configured_model_db, ConfiguredModelDB)
