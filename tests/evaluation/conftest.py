import json
from pathlib import Path

import pytest

from chap_core.database.dataset_tables import DataSetWithObservations, Observation, DataSet
from chap_core.database.tables import BackTestRead, OldBackTestRead, BackTestForecast, BackTest
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


@pytest.fixture
def dataset():
    observations = [
        Observation(
            feature_name='disease_cases',
            id=t*2+loc,
            dataset_id=1,
            period=f'2022-0{t+1}',
            org_unit=f'OrgUnit{loc+1}',
            value=float(t+loc)) for t in range(3) for loc in range(2)]
    return DataSet(
        id=1,
        name="Test Dataset",
        type="Test Type",
        geojson=None,
        covariates=[],
        observations=observations,
        created=None
    )


@pytest.fixture
def backtest(dataset):
    forecasts = [
        BackTestForecast(
            id=t*2+loc,
            backtest_id=1,
            period=f'2022-0{t+2}',
            org_unit=f'OrgUnit{loc+1}',
            last_train_period=f'2022-0{t+1}',
            last_seen_period=f'2022-0{t+1}',
            values=[float(t+loc+1), float(t+loc+2), float(t+loc+3)]
        ) for t in range(2) for loc in range(2)
    ]

    return BackTest(
        id=1,
        dataset_id=dataset.id,
        dataset=dataset,
        model_id="Test Model",
        name="Test BackTest",
        created=None,
        meta_data={},
        forecasts=forecasts,
        metrics=[]
    )
