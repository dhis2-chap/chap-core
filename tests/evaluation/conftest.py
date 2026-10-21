import json
from pathlib import Path

import pandas as pd
import pytest
import numpy as np

from chap_core.assessment.flat_representations import FlatObserved, FlatForecasts
from chap_core.assessment.evaluation import Evaluation
from chap_core.database.dataset_tables import DataSetWithObservations, Observation, DataSet
from chap_core.database.tables import (
    BacktestRead,
    OldBacktestRead,
    BacktestForecast,
    Backtest,
    BacktestMetric,
    BacktestSpecification,
)
from chap_core.simulation.naive_simulator import DatasetDimensions, AdditiveSimulator, BacktestSimulator


@pytest.fixture
def data_folder():
    return Path(__file__).parent / "data"


class BacktestOW(OldBacktestRead):
    forecasts: list[BacktestForecast]


@pytest.fixture(autouse=True)
def backtest_read(data_folder):
    read = open(data_folder / "BacktestRead.json").read()
    return BacktestOW.model_validate_json(read)


@pytest.fixture
def dataset_read(data_folder):
    read = open(data_folder / "DatasetRead.json").read()
    data = json.loads(read)
    print(data.keys())
    data["covariates"] = []
    return DataSetWithObservations.model_validate(data)


org_units = ["OrgUnit1", "OrgUnit2"]
periods = ["2022-01", "2022-02"]
periods_weeks = ["2022W01", "2022W02"]
last_seen_periods = ["2021-11", "2021-12"]
last_seen_periods_weeks = ["2021W51", "2021W52"]

# Large dataset parameters for stress testing
org_units_large = [f"OrgUnit{i + 1}" for i in range(20)]
periods_weeks_large = [f"{year}W{week:02d}" for year in range(2020, 2025) for week in range(1, 53)]
last_seen_period_weeks_large = "2019W52"


@pytest.fixture
def dataset():
    observations = [
        Observation(
            feature_name="disease_cases",
            id=t * 2 + loc,
            dataset_id=1,
            period=periods[t],
            org_unit=org_units[loc],
            value=float(t + loc),
        )
        for t in range(2)
        for loc in range(2)
    ]
    return DataSet(
        id=1,
        name="Test Dataset",
        type="Test Type",
        geojson=None,
        covariates=[],
        observations=observations,
        created=None,
    )


@pytest.fixture
def dataset_weeks():
    observations = [
        Observation(
            feature_name="disease_cases",
            id=t * 2 + loc,
            dataset_id=1,
            period=periods_weeks[t],
            org_unit=org_units[loc],
            value=float(t + loc),
        )
        for t in range(2)
        for loc in range(2)
    ]
    return DataSet(
        id=1,
        name="Test Dataset",
        type="Test Type",
        geojson=None,
        covariates=[],
        observations=observations,
        created=None,
    )


@pytest.fixture
def dataset_weeks_large():
    """Large dataset with 5 years of weekly observations for 20 org units."""
    observations = [
        Observation(
            feature_name="disease_cases",
            id=t * 20 + loc,
            dataset_id=1,
            period=periods_weeks_large[t],
            org_unit=org_units_large[loc],
            value=float(t + loc),
        )
        for t in range(len(periods_weeks_large))
        for loc in range(len(org_units_large))
    ]
    return DataSet(
        id=1,
        name="Large Test Dataset",
        type="Test Type",
        geojson=None,
        covariates=[],
        observations=observations,
        created=None,
    )


@pytest.fixture
def forecasts():
    return [
        BacktestForecast(
            id=t * 2 * 2 + loc * 2 + ls,
            backtest_id=1,
            period=f"2022-0{t + 1}",
            org_unit=f"OrgUnit{loc + 1}",
            last_train_period=last_seen_periods[ls],
            last_seen_period=last_seen_periods[ls],
            values=[float(t + loc + 1), float(t + loc + 2), float(t + loc + 3)],
        )
        for t in range(2)
        for loc in range(2)
        for ls in range(2)
    ]


@pytest.fixture
def forecasts_weeks():
    return [
        BacktestForecast(
            id=t * 2 * 2 + loc * 2 + ls,
            backtest_id=1,
            period=f"2022W0{t + 1}",
            org_unit=f"OrgUnit{loc + 1}",
            last_train_period=last_seen_periods_weeks[ls],
            last_seen_period=last_seen_periods_weeks[ls],
            values=[float(t + loc + 1), float(t + loc + 2), float(t + loc + 3)],
        )
        for t in range(2)
        for loc in range(2)
        for ls in range(2)
    ]


@pytest.fixture
def forecasts_weeks_large():
    """Large forecast fixture with 5 years of weekly data, 20 locations, 100 samples each.

    Generates approximately 5,200 forecast objects (260 weeks × 20 locations)
    with 100 samples per forecast for stress testing.
    """
    return [
        BacktestForecast(
            id=t * 20 + loc,
            backtest_id=1,
            period=periods_weeks_large[t],
            org_unit=org_units_large[loc],
            last_train_period=last_seen_period_weeks_large,
            last_seen_period=last_seen_period_weeks_large,
            values=[float(t + loc + sample) for sample in range(100)],
        )
        for t in range(len(periods_weeks_large))
        for loc in range(len(org_units_large))
    ]


@pytest.fixture
def backtest(dataset, forecasts):
    return Backtest(
        id=1,
        dataset_id=dataset.id,
        dataset=dataset,
        model_id="Test Model",
        model_db_id=1,
        specification=BacktestSpecification(dataset_id=1),
        name="Test Backtest",
        created=None,
        aggregate_metrics={},
        forecasts=forecasts,
        metrics=[],
    )


@pytest.fixture
def backtest_weeks(dataset_weeks, forecasts_weeks):
    return Backtest(
        id=1,
        dataset_id=dataset_weeks.id,
        dataset=dataset_weeks,
        model_id="Test Model",
        model_db_id=1,
        specification=BacktestSpecification(dataset_id=1),
        name="Test Backtest",
        created=None,
        aggregate_metrics={},
        forecasts=forecasts_weeks,
        metrics=[],
    )


@pytest.fixture
def backtest_weeks_large(dataset_weeks_large, forecasts_weeks_large):
    """Large backtest with 5 years of weekly forecasts for 20 org units."""
    return Backtest(
        id=1,
        dataset_id=dataset_weeks_large.id,
        dataset=dataset_weeks_large,
        model_id="Test Model Large",
        model_db_id=1,
        specification=BacktestSpecification(dataset_id=1),
        name="Large Test Backtest",
        created=None,
        aggregate_metrics={},
        forecasts=forecasts_weeks_large,
        metrics=[],
    )


@pytest.fixture
def backtest_empty(dataset):
    """Backtest with no forecasts for edge case testing."""
    return Backtest(
        id=1,
        dataset_id=dataset.id,
        dataset=dataset,
        model_id="Test Model",
        model_db_id=1,
        specification=BacktestSpecification(dataset_id=1),
        name="Empty Test Backtest",
        created=None,
        aggregate_metrics={},
        forecasts=[],
        metrics=[],
        org_units=[],
        split_periods=[],
    )


@pytest.fixture
def backtest_metrics(forecasts):
    return [
        BacktestMetric(
            id=forecast.id,
            backtest_id=forecast.backtest_id,
            metric_id="MAE",
            period=forecast.period,
            org_unit=forecast.org_unit,
            last_train_period=forecast.last_train_period,
            last_seen_period=forecast.last_seen_period,
            value=sum(forecast.values) / len(forecast.values),  # Example metric calculation
        )
        for forecast in forecasts
    ]


@pytest.fixture
def data_dims():
    dims = DatasetDimensions(
        locations=["loc1", "loc2", "loc3"],
        time_periods=[f"{year}{month:02d}" for year in ("2020", "2021", "2022") for month in range(1, 13)],
        target="disease_cases",
        features=["mean_temperature"],
    )
    return dims


@pytest.fixture
def simulated_dataset(data_dims, dummy_geojson):
    simulator = AdditiveSimulator()
    dataset = simulator.simulate(data_dims)
    dataset.geojson = dummy_geojson
    return dataset


@pytest.fixture
def simulated_backtest(simulated_dataset, data_dims):
    backtest = BacktestSimulator().simulate(simulated_dataset, data_dims)
    return backtest


@pytest.fixture
def old_backtest_file(tmp_path):
    """Creates an old format backtest file corresponding to chap-core versions <= 1.1.1 for testing backward compatibility."""
    import shutil

    shutil.copy(Path(__file__).parent / "data" / "backtest_file_version_le_1.1.1.nc", tmp_path / "tmp_backtest.nc")
    yield tmp_path / "tmp_backtest.nc"
    (tmp_path / "tmp_backtest.nc").unlink()


@pytest.fixture
def old_backtest(old_backtest_file):
    return Evaluation.from_file(old_backtest_file).to_backtest()


@pytest.fixture
def dummy_geojson():
    """Dummy GeoJSON with three disjoint polygons for testing.

    Polygons represent fictional administrative regions with:
    - Clear separation between regions for visibility
    - Irregular boundaries resembling real districts
    - Realistic lat/lon coordinates (around East Africa region)
    - Counter-clockwise winding order (GeoJSON standard)
    """
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "loc1",
                "properties": {"id": "loc1", "name": "Northern District"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [29.0, 0.0],
                            [29.0, 0.2],
                            [29.2, 0.4],
                            [29.4, 0.5],
                            [29.6, 0.3],
                            [29.5, 0.1],
                            [29.3, -0.1],
                            [29.0, 0.0],
                        ]
                    ],
                },
            },
            {
                "type": "Feature",
                "id": "loc2",
                "properties": {"id": "loc2", "name": "Central District"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [30.5, 0.0],
                            [30.8, -0.1],
                            [31.0, -0.1],
                            [31.1, -0.3],
                            [31.0, -0.5],
                            [30.7, -0.6],
                            [30.5, -0.5],
                            [30.5, 0.0],
                        ]
                    ],
                },
            },
            {
                "type": "Feature",
                "id": "loc3",
                "properties": {"id": "loc3", "name": "Southern District"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [29.5, -1.5],
                            [29.7, -1.6],
                            [30.0, -1.5],
                            [30.1, -1.5],
                            [30.1, -1.8],
                            [29.9, -2.0],
                            [29.6, -2.1],
                            [29.5, -2.0],
                            [29.5, -1.5],
                        ]
                    ],
                },
            },
        ],
    }


@pytest.fixture
def flat_forecasts():
    return FlatForecasts(
        pd.DataFrame(
            {
                "location": ["loc1", "loc1", "loc2", "loc2"],
                "time_period": ["2023-W01", "2023-W02", "2023-W01", "2023-W02"],
                "horizon_distance": [1, 2, 1, 2],
                "sample": [1, 1, 1, 1],
                "forecast": [10.0, 12.0, 21.0, 23.0],
            }
        )
    )


@pytest.fixture
def flat_observations():
    return FlatObserved(
        pd.DataFrame(
            {
                "location": ["loc1", "loc1", "loc2", "loc2"],
                "time_period": ["2023-W01", "2023-W02", "2023-W01", "2023-W02"],
                "disease_cases": [11.0, 13.0, 19.0, 21.0],
            }
        )
    )


@pytest.fixture
def flat_forecasts_multiple_samples():
    """Forecasts with multiple samples per location/time_period/horizon.

    For each location/time_period/horizon, we have 3 samples.
    The median should be used for deterministic metrics.

    loc1, 2023-W01, horizon 1: samples [8, 10, 15] -> median = 10, obs = 11, error = 1
    loc1, 2023-W02, horizon 2: samples [11, 12, 16] -> median = 12, obs = 13, error = 1
    loc2, 2023-W01, horizon 1: samples [17, 21, 24] -> median = 21, obs = 19, error = 2
    loc2, 2023-W02, horizon 2: samples [20, 23, 26] -> median = 23, obs = 21, error = 2
    """
    return FlatForecasts(
        pd.DataFrame(
            {
                "location": ["loc1"] * 3 + ["loc1"] * 3 + ["loc2"] * 3 + ["loc2"] * 3,
                "time_period": ["2023-W01"] * 3 + ["2023-W02"] * 3 + ["2023-W01"] * 3 + ["2023-W02"] * 3,
                "horizon_distance": [1] * 3 + [2] * 3 + [1] * 3 + [2] * 3,
                "sample": [0, 1, 2] * 4,
                "forecast": [8.0, 10.0, 15.0, 11.0, 12.0, 16.0, 17.0, 21.0, 24.0, 20.0, 23.0, 26.0],
            }
        )
    )


@pytest.fixture
def crps_example_data():
    observations = np.array([1.0, 2.0], dtype=float)
    forecasts = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=float,
    )
    return observations, forecasts


@pytest.fixture
def alert_history():
    """June history for location A: mean 100, std ~7.9, so a seasonal threshold of 100 + k * 7.9."""
    values = [90.0, 95.0, 100.0, 105.0, 110.0]
    return pd.DataFrame(
        {"location": "A", "time_period": [f"{year}-06" for year in range(2018, 2023)], "disease_cases": values}
    )


@pytest.fixture
def alert_levels():
    """Two-level ladder on the seasonal channel: monitor above the mean, action above mean + 1 std."""
    from chap_core.assessment.thresholds.params import SeasonalParams
    from chap_core.database.alert_tables import AlertLevel

    return [
        AlertLevel(
            name="monitor",
            threshold_params=SeasonalParams(type="seasonal", std_multiplier=0.0),
            exceedance_threshold=0.5,
        ),
        AlertLevel(
            name="action",
            threshold_params=SeasonalParams(type="seasonal", std_multiplier=1.0),
            exceedance_threshold=0.5,
        ),
    ]


@pytest.fixture
def alert_observations():
    """2023-06 breaches monitor only; 2024-06 breaches nothing."""
    return pd.DataFrame({"location": "A", "time_period": ["2023-06", "2024-06"], "disease_cases": [104.0, 99.0]})


@pytest.fixture
def alert_forecasts():
    """Samples predicting monitor for 2023-06 at horizon 1, nothing at horizon 2, and action for 2024-06."""
    cells = [
        ("2023-06", 1, [101.0, 102.0, 103.0]),
        ("2023-06", 2, [95.0, 96.0, 97.0]),
        ("2024-06", 1, [110.0, 111.0, 112.0]),
    ]
    return pd.DataFrame(
        [
            {"location": "A", "time_period": period, "horizon_distance": horizon, "sample": i, "forecast": value}
            for period, horizon, samples in cells
            for i, value in enumerate(samples)
        ]
    )
