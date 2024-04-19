from climate_health.cli import get_full_dataframe
from climate_health.external.models.jax_models.model_spec import SSMForecasterNuts, NutsParams
from climate_health.external.models.jax_models.specs import SSMWithoutWeather
import pytest

from climate_health.time_period import Month


@pytest.fixture
def dhis_process():
    base_url = 'https://play.dhis2.org/40.3.0/'
    username = 'admin'
    password = 'district'
    from climate_health.dhis2_interface.ChapProgram import ChapPullPost
    process = ChapPullPost(dhis2Baseurl=base_url.rstrip('/'), dhis2Username=username, dhis2Password=password)
    return process


def test_pull_from_play(dhis_process):
    process = dhis_process
    full_data_frame = get_full_dataframe(process)


@pytest.fixture()
def dhis_test_model(data_path):
    return SSMForecasterNuts.load(data_path / 'dhis_test_model')


def test_push(dhis_process, dhis_test_model):
    process = dhis_process
    full_data_frame = get_full_dataframe(process)
    modelspec = SSMWithoutWeather()
    model = dhis_test_model
    predictions = model.prediction_summary(Month(full_data_frame.end_timestamp))
    process.pushDataToDHIS2(predictions, modelspec.__class__.__name__)


# Skipped because failing
def test_flow(dhis_process):
    process = dhis_process
    full_data_frame = get_full_dataframe(process)
    modelspec = SSMWithoutWeather()
    model = SSMForecasterNuts(modelspec, NutsParams(n_samples=10, n_warmup=10))
    model.train(full_data_frame)
    predictions = model.prediction_summary(Month(full_data_frame.end_timestamp))
    print(process.pushDataToDHIS2(predictions, modelspec.__class__.__name__))
