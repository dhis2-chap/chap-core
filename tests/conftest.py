import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, List
from unittest.mock import patch

# ignore showing plots in tests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlmodel import SQLModel

from chap_core import util
from chap_core.api_types import RequestV1
from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.database.dataset_tables import DataSet, ObservationBase
from chap_core.datatypes import HealthPopulationData, SimpleClimateData
from chap_core.geometry import Polygons
from chap_core.rest_api.data_models import FetchRequest
from chap_core.rest_api.v1.routers.crud import DatasetCreate, PredictionCreate
from chap_core.rest_api.worker_functions import WorkerConfig

from .data_fixtures import *

# Don't use pytest-celery if on windows
IS_WINDOWS = os.name == "nt"

# logger
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def redis_available():
    if not util.redis_available():
        pytest.skip("Redis not available")


pytest_plugins = ("celery.contrib.pytest",)


@pytest.fixture(scope="session")
def celery_config(database_url):
    """Overrides the default redis broker for all celery worker tests based on env vars"""
    logger.debug(f"Using celery database_url: {database_url}")
    host = os.getenv("REDIS_HOST", "localhost")  # default to localhost for backwards compatibility
    port = os.getenv("REDIS_PORT", "6379")
    logger.debug(f"Using redis host: {host}:{port}")
    return {
        "broker_url": f"redis://{host}:{port}",  # /0",
        "result_backend": f"redis://{host}:{port}",  # /1",
        "task_serializer": "pickle",
        "accept_content": ["pickle"],
        "result_serializer": "pickle",
        "database_url": database_url,
    }


@pytest.fixture(scope="session")
def celery_session_worker(redis_available, celery_session_worker):
    return celery_session_worker


@pytest.fixture(scope="session")
def celery_worker_pool():
    return "prefork"


# if not IS_WINDOWS:
#
# else:
#     @pytest.fixture(scope='session')
#     def celery_session_worker():
#         pytest.skip("pytest-celery not available on Windows")

plt.ion()


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False, help="Run slow tests")


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
    path = Path(__file__).parent.parent / "target" / "test_outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def models_path():
    return Path(__file__).parent.parent / "external_models"


@pytest.fixture
def local_data_path():
    path = Path("/home/knut/Data/ch_data/")
    if not path.exists():
        pytest.skip("Data path does not exist")
    return path


@pytest.fixture
def tests_path():
    return Path(__file__).parent


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
    train, tests = train_test_generator(weekly_full_data, prediction_length=12)
    training_path = tmp_path / "training_data.csv"
    train.to_csv(training_path)
    historic, masked, _ = next(tests)
    historic_path = tmp_path / "historic_data.csv"
    historic.to_csv(historic_path)
    future_path = tmp_path / "future_data.csv"
    masked.to_csv(future_path)
    return training_path, historic_path, future_path


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
    dicts["estimator_id"] = "naive_model"
    return json.dumps(dicts)


@pytest.fixture
def laos_request_2(local_data_path):
    filepath = local_data_path / "laos_request_2.json"
    with open(filepath, "r") as f:
        text = f.read()
    dicts = json.loads(text)
    dicts["estimator_id"] = "naive_model"
    return json.dumps(dicts)


@pytest.fixture
def laos_request_3(local_data_path):
    filepath = local_data_path / "laos_request_3.json"
    with open(filepath, "r") as f:
        text = f.read()
    dicts = json.loads(text)
    dicts["estimator_id"] = "naive_model"
    return json.dumps(dicts)


@pytest.fixture
def dataset_create(big_request_json):
    data = RequestV1.model_validate_json(big_request_json)
    return DatasetCreate(
        name="test",
        type="evaluation",
        geojson=data.orgUnitsGeoJson.model_dump(),
        observations=[
            ObservationBase(
                feature_name=f.featureId if f.featureId != "diseases" else "disease_cases",
                period=d.pe,
                orgUnit=d.ou,
                value=d.value,
            )
            for f in data.features
            for d in f.data
        ],
    )


@pytest.fixture()
def example_polygons(data_path):
    return Polygons.from_file(data_path / "example_polygons.geojson").data


@pytest.fixture
def make_prediction_request(dataset_create):
    return PredictionCreate(model_id="naive_model", metaData={"test": "test2"}, **dataset_create.dict())


# @pytest.fixture
# def celery_app():
#     app = Celery(
#         broker="memory://",
#         backend="cache+memory://",
#         include=['chap_core.rest_api.celery_tasks']
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
    def __init__(self, *args, **kwargs): ...

    def get_historical_era5(self, features, periodes, fetch_requests: Optional[List[FetchRequest]] = None):
        locations = [f["id"] for f in features["features"]]
        return DataSet(
            {
                location: SimpleClimateData(periodes, np.random.rand(len(periodes)), np.random.rand(len(periodes)))
                for location in locations
            }
        )


@pytest.fixture(scope="session")
def database_url():
    # Use target directory for test database
    project_root = Path(__file__).parent.parent
    db_dir = project_root / "target"
    db_dir.mkdir(exist_ok=True)
    return f"sqlite:///{db_dir}/test.db"


@pytest.fixture(scope="session")
def clean_engine(database_url):
    # TODO: rename clean_engine_with_models?
    # TODO: maybe use the on_startup function instead of manually setting up things?
    engine = create_engine(database_url, connect_args={"check_same_thread": False})
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)
    from sqlmodel import Session

    from chap_core.database.model_template_seed import seed_configured_models_from_config_dir

    with Session(engine) as session:
        seed_configured_models_from_config_dir(session, skip_chapkit_models=True)
    return engine


@pytest.fixture
def test_config():
    return WorkerConfig(is_test=True)
