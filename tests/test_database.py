import pytest
from sqlalchemy import create_engine
from sqlmodel import SQLModel, Session, select
import logging

from chap_core.database.model_template_seed import add_model_template_from_url, add_configured_model, \
    seed_configured_models_from_config_dir
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
from chap_core.database.model_spec_tables import seed_with_session_wrapper
from chap_core.database.model_templates_and_config_tables import ModelTemplateMetaData, ModelTemplateDB, \
    ConfiguredModelDB, ModelConfiguration


logger = logging.getLogger(__name__)


template_urls = [
    'https://github.com/sandvelab/monthly_ar_model@7c40890df749506c72748afda663e0e1cde4e36a',
    'https://github.com/knutdrand/weekly_ar_model@15cc39068498a852771c314e8ea989e6b555b8a5',
    'https://github.com/dhis2-chap/chap_auto_ewars@0c41b1d9bd187521e62c58d581e6f5bd5127f7b5',
    'https://github.com/dhis2-chap/chap_auto_ewars_weekly@51c63a8581bc29bdb40e788a83f701ed30cca83f',
]


@pytest.fixture
def engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def engine_with_dataset(engine, weekly_full_data):
    with SessionWrapper(engine) as session:
        session.add_dataset('full_data', weekly_full_data, 'polygons')
    return engine


def test_dataset_roundrip(health_population_data, engine):
    with SessionWrapper(engine) as session:
        dataset_id = session.add_dataset('health_population', health_population_data, 'polygons')
        dataset = session.get_dataset(dataset_id, HealthPopulationData)
        assert_dataset_equal(dataset, health_population_data)


@pytest.mark.skip('Needs to seed models for this test to work')
def test_backtest(engine_with_dataset):
    with Session(engine_with_dataset) as session:
        dataset_id = session.exec(select(DataSet.id)).first()
    # with patch('chap_core.database.database.engine', engine_with_dataset):
    with SessionWrapper(engine_with_dataset) as session:
        res = run_backtest(BackTestCreate(model_id='naive_model', dataset_id=dataset_id), 12, 2, 1, session=session)
    # res = run_backtest('naive_model', dataset_id, 12, 2, 1)
    with Session(engine_with_dataset) as session:
        backtests = session.exec(select(BackTest)).all()
        assert len(backtests) == 1
        backtest = backtests[0]
        assert backtest.dataset_id == dataset_id
        assert len(backtest.forecasts) == 12 * 2 * 10


@pytest.mark.skip('Needs to seed models for this test to work')
def test_add_predictions(engine_with_dataset):
    with SessionWrapper(engine_with_dataset) as session:
        run_prediction('naive_model', 1, 3, name='testing', metadata='', session=session)


@pytest.fixture
def model_template_yaml_config():
    return ModelTemplateConfigV2(
        name='test_model',
        required_covariates=['rainfall', 'mean_temperature'],
        allow_free_additional_continuous_covariates=False,
        user_options={},
        meta_data=ModelTemplateMetaData(author='chap_temp',
                                        author_note='Testing author note',
                                        author_assessed_status='green',
                                        description='my model',
                                        display_name='My Model'),
        entry_points=EntryPointConfig(train=CommandConfig(command='train', parameters={'param1': 'value1'}),
                                      predict=CommandConfig(command='predict', parameters={'param2': 'value2'})),
        docker_env=DockerEnvConfig(image='my_docker_image')
    )


def test_add_model_template_from_yaml_config(model_template_yaml_config, engine):
    with SessionWrapper(engine) as session:
        id = session.add_model_template_from_yaml_config(model_template_yaml_config)
        model_template = session.get_model_template(id)
        assert model_template.name == model_template_yaml_config.name
        assert model_template.required_covariates == model_template_yaml_config.required_covariates
        assert model_template.allow_free_additional_continuous_covariates == model_template_yaml_config.allow_free_additional_continuous_covariates
        assert model_template.user_options == model_template_yaml_config.user_options
        assert model_template.author_assessed_status == model_template_yaml_config.meta_data.author_assessed_status


@pytest.mark.parametrize('url', template_urls)
def test_add_model_template_from_url(engine, url):
    # url = 'https://github.com/sandvelab/monthly_ar_model@7c40890df749506c72748afda663e0e1cde4e36a'
    with SessionWrapper(engine) as session:
        template_id = add_model_template_from_url(url, session)
        configured_model_id = add_configured_model(
            template_id,
            ModelConfiguration(user_option_values={}),
            'default',
            session
        )
        external_model = session.get_configured_model_with_code(configured_model_id)
        assert isinstance(external_model, ExternalModel)


def test_seed_configured_models(engine):
    # make sure is clean
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)
    # seed with models
    with Session(engine) as session:
        # ensure db doesnt contain any models
        configured_models = session.exec(
            select(ConfiguredModelDB)
        ).all()
        assert not configured_models
        # seed with models
        seed_configured_models_from_config_dir(session)
        # seed again to check that repeated inserts are handled nicely
        seed_configured_models_from_config_dir(session)
    # test that models have been added
    with Session(engine) as session:
        configured_models = session.exec(
            select(ConfiguredModelDB)
            .join(ConfiguredModelDB.model_template)
        ).all()
        logger.info(f'A total of {len(configured_models)} configured models have been added to the db:')
        for m in configured_models:
            logger.info(f'--> {m}')
        assert len(configured_models) > 1
        model_names = [m.name for m in configured_models]
        assert 'naive_model' in model_names
