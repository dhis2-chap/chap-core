from climate_health.external.models.flax_models.flax_model import FlaxModel
from tests.external.util import check_model


def test_training(full_train_data, random_key, test_data):
    model_class = FlaxModel(rng_key=random_key, n_iter=10)
    check_model(full_train_data, model_class, random_key, test_data)
