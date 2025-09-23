from chap_core.database.dataset_tables import DataSet
from .db_fixtures import seeded_session, p_seeded_engine, base_engine, seeded_database_url
from .data_fixtures import dataset, dataset_observations, geojson, feature_names, org_units, seen_periods, prediction
from chap_core.rest_api.data_models import DatasetMakeRequest


def test_dataset(seeded_session):
    dataset = seeded_session.query(DataSet)
    assert dataset[0].data_sources[0].covariate == 'mean_temperature'
    assert dataset.count() == 1
