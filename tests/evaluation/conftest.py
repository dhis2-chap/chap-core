import json
from pathlib import Path

import pytest

from chap_core.database.dataset_tables import DataSetWithObservations
from chap_core.database.tables import BackTestRead, OldBackTestRead, BackTestForecast
from chap_core.rest_api.v1.routers.crud import DataSetRead


@pytest.fixture
def data_folder():
    return Path(__file__).parent / "data"

class BacktestOW(OldBackTestRead):
    forecasts: list[BackTestForecast]


@pytest.fixture(autouse=True)
def backtest_read(data_folder):
    read = open(data_folder / "BacktestRead.json").read()
    return BacktestOW.model_validate_json(read)

@pytest.fixture
def dataset_read(data_folder):
    read = open(data_folder / "DatasetRead.json").read()
    data = json.loads(read)
    print(data.keys())
    data['covariates'] = []
    return DataSetWithObservations.model_validate(data)

