from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB, ModelTemplateDB
from chap_core.predictor.naive_estimator import NaiveEstimator
from chap_core.testing.estimators import sanity_check_estimator


def test_train():
    estimator = NaiveEstimator()
    sanity_check_estimator(estimator)


def test_model_metadata_class_attributes():
    assert isinstance(NaiveEstimator.model_template_db, ModelTemplateDB)
    assert isinstance(NaiveEstimator.configured_model_db, ConfiguredModelDB)
