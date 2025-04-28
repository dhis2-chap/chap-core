import json
import os
import shutil
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pytest
from sqlmodel import SQLModel
from chap_core.api_types import RequestV1
from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.database.database import SessionWrapper
from chap_core.database.dataset_tables import ObservationBase, DataSet
from chap_core.database.model_spec_tables import seed_with_session_wrapper
from chap_core.database.tables import *
import pandas as pd

from chap_core.datatypes import HealthPopulationData, SimpleClimateData
from chap_core.geometry import Polygons
from chap_core.rest_api_src.data_models import FetchRequest
from chap_core.rest_api_src.v1.routers.crud import DatasetCreate, PredictionCreate
from chap_core.rest_api_src.v1.routers.analytics import MakePredictionRequest
from chap_core.rest_api_src.worker_functions import WorkerConfig
from chap_core.services.cache_manager import get_cache
from .data_fixtures import *

# ignore showing plots in tests
import matplotlib.pyplot as plt

# Don't use pytest-celery if on windows
IS_WINDOWS = os.name == "nt"


@pytest.fixture(scope='session')
def redis_available():
    import redis
    try:
        redis.Redis().ping()
        return True
    except redis.exceptions.ConnectionError:
        pytest.skip("Redis not available")


pytest_plugins = ("celery.contrib.pytest",)


@pytest.fixture(scope='session')
def celery_session_worker(redis_available, celery_session_worker):
    return celery_session_worker


# if not IS_WINDOWS:
#
# else:
#     @pytest.fixture(scope='session')
#     def celery_session_worker():
#         pytest.skip("pytest-celery not available on Windows")

plt.ion()


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="Run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark a test as a slow test.")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip_integration = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_integration)


@pytest.fixture
def data_path():
    return Path(__file__).parent.parent / "example_data"


@pytest.fixture
def output_path():
    path = Path(__file__).parent / "test_outputs"
    path.mkdir(exist_ok=True)
    return path


@pytest.fixture
def models_path():
    return Path(__file__).parent.parent / "external_models"


@pytest.fixture
def local_data_path():
    path = Path('/home/knut/Data/ch_data/')
    if not path.exists():
        pytest.skip("Data path does not exist")
    return path


@pytest.fixture
def tests_path():
    return Path(__file__).parent


@pytest.fixture(scope="session", autouse=True)
def use_test_cache():
    os.environ["TEST_ENV"] = "true"
    yield
    del os.environ["TEST_ENV"]
    cache = get_cache()
    cache.close()
    shutil.rmtree(cache.directory, ignore_errors=True)


@pytest.fixture()
def health_population_data(data_path):
    file_name = (data_path / "health_population_data").with_suffix(".csv")
    return DataSet.from_pandas(pd.read_csv(file_name), HealthPopulationData)


@pytest.fixture
def nicaragua_path(data_path):
    return (data_path / "nicaragua_weekly_data").with_suffix(".csv")


@pytest.fixture()
def weekly_full_data(nicaragua_path):
    file_name = nicaragua_path
    return DataSet.from_pandas(pd.read_csv(file_name), FullData)

@pytest.fixture
def dumped_weekly_data_paths(weekly_full_data, tmp_path):
    train, tests = train_test_generator(weekly_full_data,
                                        prediction_length=12)
    training_path = tmp_path / 'training_data.csv'
    train.to_csv(training_path)
    historic, masked, _ = next(tests)
    historic_path = tmp_path / 'historic_data.csv'
    historic.to_csv(historic_path)
    future_path = tmp_path / 'future_data.csv'
    masked.to_csv(future_path)
    return training_path, historic_path, future_path




@pytest.fixture()
def google_earth_engine():
    from chap_core.google_earth_engine.gee_era5 import GoogleEarthEngine

    try:
        return GoogleEarthEngine()
    except:
        pytest.skip("Google Earth Engine not available")


@pytest.fixture()
def mocked_gee(gee_mock):
    with patch('chap_core.rest_api_src.worker_functions.Era5LandGoogleEarthEngine', gee_mock):
        yield


@pytest.fixture()
def gee_mock():
    return GEEMock


@pytest.fixture
def request_json(data_path):
    return open(data_path / "v1_api/request.json", "r").read()


@pytest.fixture
def big_request_json(data_path):
    filepath = data_path / "anonymous_chap_request.json"
    assert os.path.exists(filepath)
    if not os.path.exists(filepath):
        pytest.skip()
    with open(filepath, "r") as f:
        return f.read()


@pytest.fixture
def laos_request(local_data_path):
    filepath = local_data_path / "laos_requet.json"
    with open(filepath, "r") as f:
        text = f.read()
    dicts = json.loads(text)
    dicts['estimator_id'] = 'naive_model'
    return json.dumps(dicts)


@pytest.fixture
def laos_request_2(local_data_path):
    filepath = local_data_path / "laos_request_2.json"
    with open(filepath, "r") as f:
        text = f.read()
    dicts = json.loads(text)
    dicts['estimator_id'] = 'naive_model'
    return json.dumps(dicts)


@pytest.fixture
def laos_request_3(local_data_path):
    filepath = local_data_path / "laos_request_3.json"
    with open(filepath, "r") as f:
        text = f.read()
    dicts = json.loads(text)
    dicts['estimator_id'] = 'naive_model'
    return json.dumps(dicts)


@pytest.fixture
def dataset_create(big_request_json):
    data = RequestV1.model_validate_json(big_request_json)
    return DatasetCreate(name='test',
                         type='evaluation',
                         geojson=data.orgUnitsGeoJson.model_dump(),
                         observations=[ObservationBase(
                             feature_name=f.featureId if f.featureId != 'diseases' else 'disease_cases',
                             period=d.pe, orgUnit=d.ou,
                             value=d.value) for f in data.features for d in f.data])


@pytest.fixture()
def example_polygons(data_path):
    return Polygons.from_file(data_path / "example_polygons.geojson").data


@pytest.fixture
def make_prediction_request(dataset_create):
    return PredictionCreate(model_id='naive_model',
                            metaData={'test': 'test2'},
                            **dataset_create.dict())


# @pytest.fixture
# def celery_app():
#     app = Celery(
#         broker="memory://",
#         backend="cache+memory://",
#         include=['chap_core.rest_api_src.celery_tasks']
#     )
#     #app.conf.task_always_eager = True  # Run tasks synchronously
#     #app.conf.result_backend = "cache+memory://"
#     app.conf.update(
#         task_always_eager=True,
#         task_eager_propagates=True,
#         task_serializer="pickle",
#         accept_content=["pickle"],  # Allow pickle serialization
#         result_serializer="pickle",
#     )
#     return app

class GEEMock:
    def __init__(self, *args, **kwargs):
        ...

    def get_historical_era5(self, features, periodes, fetch_requests: Optional[List[FetchRequest]] = None):
        locations = [f['id'] for f in features['features']]
        return DataSet({location:
                            SimpleClimateData(periodes, np.random.rand(len(periodes)),
                                              np.random.rand(len(periodes)))
                        for location in locations})


@pytest.fixture(scope='session')
def database_url():
    cur_dir = Path(__file__).parent
    return f'sqlite:///{cur_dir}/test.db'


@pytest.fixture(scope='session')
def clean_engine(database_url):
    engine = create_engine(database_url,
                           connect_args={"check_same_thread": False})
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)
    with SessionWrapper(engine) as session:
        seed_with_session_wrapper(session)
    return engine


@pytest.fixture(scope='session')
def celery_config(database_url):
    print(f"Using database_url: {database_url}")
    return {
        'broker_url': 'redis://localhost:6379',
        'result_backend': 'redis://localhost:6379',
        'task_serializer': 'pickle',
        'accept_content': ['pickle'],
        'result_serializer': 'pickle',
        'database_url': database_url,
    }


@pytest.fixture(scope='session')
def celery_worker_pool():
    return 'prefork'


@pytest.fixture
def test_config():
    return WorkerConfig(is_test=True)
