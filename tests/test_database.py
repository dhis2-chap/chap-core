import pytest
from sqlalchemy import create_engine
from sqlmodel import SQLModel, Session, select

from chap_core.database.model_template_seed import add_model_template_from_url, add_configured_model, \
    seed_configured_models
from chap_core.database.tables import BackTest
from chap_core.database.dataset_tables import DataSet
from chap_core.datatypes import HealthPopulationData
from chap_core.external.model_configuration import ModelInfo, ModelTemplateConfig, ModelTemplateConfigV2, \
    EntryPointConfig, CommandConfig, DockerEnvConfig
from chap_core.models.external_model import ExternalModel
from chap_core.models.model_template import ExternalModelTemplate
from chap_core.rest_api_src.db_worker_functions import run_backtest, run_prediction
from chap_core.rest_api_src.data_models import BackTestCreate
from chap_core.testing.testing import assert_dataset_equal
from chap_core.database.database import SessionWrapper
import chap_core.database.database
from chap_core.database.model_spec_tables import seed_with_session_wrapper, ModelTemplateMetaData
from chap_core.database.model_template_seed import template_urls


@pytest.fixture
def engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def seeded_engine(engine, weekly_full_data):
    with SessionWrapper(engine) as session:
        session.add_dataset('full_data', weekly_full_data, 'polygons')
    return engine


def test_dataset_roundrip(health_population_data, engine):
    with SessionWrapper(engine) as session:
        dataset_id = session.add_dataset('health_population', health_population_data, 'polygons')
        dataset = session.get_dataset(dataset_id, HealthPopulationData)
        assert_dataset_equal(dataset, health_population_data)


@pytest.mark.slow
def test_backtest(seeded_engine):
    with Session(seeded_engine) as session:
        dataset_id = session.exec(select(DataSet.id)).first()
    # with patch('chap_core.database.database.engine', seeded_engine):
    with SessionWrapper(seeded_engine) as session:
        res = run_backtest(BackTestCreate(model_id='naive_model', dataset_id=dataset_id), 12, 2, 1, session=session)
    # res = run_backtest('naive_model', dataset_id, 12, 2, 1)
    with Session(seeded_engine) as session:
        backtests = session.exec(select(BackTest)).all()
        assert len(backtests) == 1
        backtest = backtests[0]
        assert backtest.dataset_id == dataset_id
        assert len(backtest.forecasts) == 12 * 2 * 10


@pytest.mark.slow
def test_add_predictions(seeded_engine):
    with SessionWrapper(seeded_engine) as session:
        run_prediction('naive_model', 1, 3, name='testing', metadata='', session=session)


@pytest.mark.skip
def test_seed(seeded_engine):
    with SessionWrapper(seeded_engine) as session:
        seed_with_session_wrapper(session)
        seed_with_session_wrapper(session)


@pytest.fixture
def model_template_config():
    return ModelTemplateConfigV2(
        name='test_model',
        required_covariates=['rainfall', 'mean_temperature'],
        allow_free_additional_continuous_covariates=False,
        user_options={},
        meta_data=ModelTemplateMetaData(author='chap_temp',
                                        description='my model',
                                        display_name='My Model'),
        entry_points=EntryPointConfig(train=CommandConfig(command='train', parameters={'param1': 'value1'}),
                                      predict=CommandConfig(command='predict', parameters={'param2': 'value2'})),
        docker_env=DockerEnvConfig(image='my_docker_image')
    )


def test_add_model_template(model_template_config, engine):
    with SessionWrapper(engine) as session:
        id = session.add_model_template(model_template_config)
        model_template = session.get_model_template(id)
        assert model_template.name == model_template_config.name
        assert model_template.required_covariates == model_template_config.required_covariates
        assert model_template.allow_free_additional_continuous_covariates == model_template_config.allow_free_additional_continuous_covariates
        assert model_template.user_options == model_template_config.user_options


# @pytest.mark.skip
@pytest.mark.parametrize('url', template_urls)
def test_add_model_template_from_url(engine, url):
    # url = 'https://github.com/sandvelab/monthly_ar_model@7c40890df749506c72748afda663e0e1cde4e36a'
    with SessionWrapper(engine) as session:
        template_id = add_model_template_from_url(url, session)
        configured_model_id = add_configured_model(
            template_id, {}, session)
        external_model = session.get_configured_model(configured_model_id)
        assert isinstance(external_model, ExternalModel)


def test_seed_configured_models(engine):
    with SessionWrapper(engine) as session:
        seed_configured_models(session)
