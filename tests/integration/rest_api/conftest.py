import datetime
import typing

import numpy as np
import pytest
from pydantic_geojson import PointModel
from sqlmodel import select, Session

from chap_core.api_types import FeatureCollectionModel
from chap_core.database.dataset_tables import DataSet, Observation, DataSource
from chap_core.database.tables import Prediction, BackTest, BackTestForecast, BackTestMetric, PredictionSamplesEntry
from chap_core.rest_api.v1.rest_api import app
from chap_core.rest_api.v1.routers.analytics import BackTestParams
from chap_core.rest_api.v1.routers.dependencies import get_session


@pytest.fixture
def feature_names():
    return ["mean_temperature", "rainfall", "population"]


@pytest.fixture
def seen_periods():
    return [f"{year}-{month:02d}" for year in range(2020, 2023) for month in range(1, 13)]


@pytest.fixture
def future_periods():
    return [f"2023-{month:02d}" for month in range(1, 3)]


@pytest.fixture
def org_units():
    return ["loc_1", "loc_2", "loc_3"]


@pytest.fixture
def backtest_params():
    return BackTestParams(n_periods=3, n_splits=2, stride=1)


@pytest.fixture
def geojson(org_units) -> FeatureCollectionModel:
    return FeatureCollectionModel(
        features=[
            {"type": "Feature", "id": ou, "properties": {"name": ou}, "geometry": PointModel(coordinates=[0.0, 0.0])}
            for ou in org_units
        ]
    )


@pytest.fixture
def seen_periods_weekly():
    # 2020 has 53 weeks (leap year starting on Wednesday), 2021 and 2022 have 52 weeks
    import datetime

    periods = []
    for year in range(2020, 2023):
        # Check how many weeks the year has using ISO calendar
        last_day = datetime.date(year, 12, 31)
        max_week = last_day.isocalendar()[1]
        # If Dec 31 is in week 1 of next year, then max_week should be obtained from Dec 28
        if max_week == 1:
            max_week = datetime.date(year, 12, 28).isocalendar()[1]
        periods.extend([f"{year}W{week:02d}" for week in range(1, max_week + 1)])
    return periods


@pytest.fixture
def dataset_observations(feature_names: list[str], org_units: list[str], seen_periods: list[str]) -> list[Observation]:
    return _generate_observations(feature_names, org_units, seen_periods)


def _generate_observations(
    feature_names: list[str], org_units: list[str], seen_periods: list[str]
) -> list[Observation]:
    observations = [
        Observation(org_unit=ou, feature_name=fn, period=tp, value=float(ou_id + np.sin(t % 12) / 2))
        for ou_id, ou in enumerate(org_units)
        for fn in (feature_names + ["disease_cases"])
        for t, tp in enumerate(seen_periods)
    ]
    return observations


@pytest.fixture
def dataset_observations_weekly(
    feature_names: list[str], org_units: list[str], seen_periods_weekly: list[str]
) -> list[Observation]:
    observations = [
        Observation(org_unit=ou, feature_name=fn, period=tp, value=float(ou_id + np.sin(t % 52) / 2))
        for ou_id, ou in enumerate(org_units)
        for fn in (feature_names + ["disease_cases"])
        for t, tp in enumerate(seen_periods_weekly)
    ]
    return observations


@pytest.fixture
def dataset(org_units, feature_names, seen_periods, dataset_observations, geojson):
    return _make_dataset(dataset_observations, feature_names, geojson, org_units, seen_periods)


def _make_dataset(
    dataset_observations: list[Observation],
    feature_names: list[str],
    geojson: FeatureCollectionModel,
    org_units: list[str],
    seen_periods: list[str],
    name="testing dataset",
) -> DataSet:
    return DataSet(
        name=name,
        geojson=geojson.model_dump_json(),
        observations=dataset_observations,
        covariates=feature_names + ["disease_cases"],
        created=datetime.datetime.now(),
        data_sources=[DataSource(covariate=fn, data_element_id=f"de_{i}") for i, fn in enumerate(feature_names)],
        first_period=seen_periods[0],
        last_period=seen_periods[-1],
        org_units=org_units,
        period_type="month",
    )


@pytest.fixture
def dataset_with_nans(org_units, feature_names, seen_periods, geojson):
    dataset_observations = _generate_observations(feature_names, org_units, seen_periods)
    dc = [obs for obs in dataset_observations if obs.feature_name == "disease_cases"]
    assert len(dc)
    for obs in dc[3:]:
        obs.value = float(np.nan)

    ds = _make_dataset(
        dataset_observations,
        feature_names,
        geojson,
        org_units,
        seen_periods,
        name="dataset_with_nans",
    )
    # Introduce some NaN values
    return ds


@pytest.fixture
def dataset_wo_meta_observations(
    feature_names: list[str], org_units: list[str], seen_periods: list[str]
) -> list[Observation]:
    observations = [
        Observation(org_unit=ou, feature_name=fn, period=tp, value=float(ou_id + np.sin(t % 12) / 2))
        for ou_id, ou in enumerate(org_units)
        for fn in (feature_names + ["disease_cases"])
        for t, tp in enumerate(seen_periods)
    ]
    return observations


@pytest.fixture
def dataset_wo_meta(org_units, feature_names, seen_periods, dataset_wo_meta_observations, geojson):
    return DataSet(
        name="incomplete_dataset",
        geojson=geojson.model_dump_json(),
        observations=dataset_wo_meta_observations,
        covariates=feature_names + ["disease_cases"],
        created=datetime.datetime.now(),
        first_period=seen_periods[0],
        last_period=seen_periods[-1],
    )


@pytest.fixture
def predictions(future_periods, org_units):
    return [
        PredictionSamplesEntry(period=tp, org_unit=ou, values=[float(tp_id + 0.1 * s) for s in range(10)])
        for tp_id, tp in enumerate(future_periods)
        for ou in org_units
    ]


@pytest.fixture
def prediction(dataset, predictions):
    return Prediction(
        model_id="naive_model",
        model_db_id=1,
        n_periods=3,
        name="test prediction",
        created=datetime.datetime.now(),
        forecasts=predictions,
        dataset=dataset,
    )


@pytest.fixture
def forecasts(seen_periods, org_units, backtest_params):
    forecasts = _generate_forecasts(backtest_params, org_units, seen_periods)
    return forecasts


@pytest.fixture
def forecasts_2(seen_periods, org_units, backtest_params):
    forecasts = _generate_forecasts(backtest_params, org_units, seen_periods)
    return forecasts


def _generate_forecasts(
    backtest_params: BackTestParams, org_units: list[str], seen_periods: list[str]
) -> list[typing.Any]:
    start_split = len(seen_periods) - backtest_params.n_splits - backtest_params.n_periods
    forecasts = []
    for start in range(start_split, start_split + backtest_params.n_splits):
        forecasts.extend(
            [
                BackTestForecast(
                    org_unit=ou,
                    last_train_period=seen_periods[start],
                    last_seen_period=seen_periods[start],
                    period=seen_periods[start + t],
                    values=[float(t + 0.1 * p) for p in range(10)],
                )
                for t in range(backtest_params.n_periods)
                for ou in org_units
            ]
        )
    true_forecasts = backtest_params.n_splits * backtest_params.n_periods * len(org_units)
    assert len(forecasts) == true_forecasts, f"Expected {true_forecasts} forecasts, got {len(forecasts)}"
    return forecasts


@pytest.fixture
def backtest(dataset, forecasts):
    return BackTest(
        name="test backtest",
        dataset=dataset,
        forecasts=forecasts,
        model_id="naive_model",
        aggregate_metrics={"MAE": 1.5},
        model_db_id=1,
    )


@pytest.fixture
def backtest_with_nans(dataset_with_nans, forecasts_2):
    return BackTest(
        name="test_backtest_with_nans",
        dataset=dataset_with_nans,
        forecasts=forecasts_2,
        model_id="naive_model",
        aggregate_metrics={"MAE": 1.5},
        model_db_id=1,
    )


# Database fixtures
@pytest.fixture
def seeded_database_url(tmp_path):
    db_path = tmp_path / "seeded_db.sqlite"
    return f"sqlite:///{db_path}"


@pytest.fixture
def base_engine(seeded_database_url):
    from sqlalchemy import create_engine
    from sqlmodel import SQLModel, Session

    engine = create_engine(seeded_database_url, connect_args={"check_same_thread": False})
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)

    from chap_core.database.model_template_seed import seed_configured_models_from_config_dir

    with Session(engine) as session:
        seed_configured_models_from_config_dir(session, skip_chapkit_models=True)
    return engine


@pytest.fixture
def p_seeded_engine(
    base_engine,
    prediction,
    backtest,
    dataset_wo_meta,
    dataset,
    dataset_observations,
    dataset_with_nans,
    backtest_with_nans,
):
    from sqlmodel import Session

    with Session(base_engine) as session:
        # Don't add observations separately - they will be added via the datasets
        session.add(prediction)
        session.add(backtest)
        session.add(dataset)
        session.add(dataset_wo_meta)
        session.add(dataset_with_nans)
        session.add(backtest_with_nans)
        session.commit()
        session.refresh(prediction)
        all_observations = session.exec(select(Observation)).all()
        for obs in all_observations:
            print(obs)
        d_observations = list(session.exec(select(DataSet).where(DataSet.name == "testing dataset")).one().observations)
        assert d_observations, d_observations
        observations = list(
            session.exec(select(BackTest).where(BackTest.name == "test backtest")).one().dataset.observations
        )
        obs_with_nan = list(
            session.exec(select(BackTest).where(BackTest.name == "test_backtest_with_nans")).one().dataset.observations
        )
        assert any(obs.value is None for obs in obs_with_nan), "No NaN observations found in dataset_with_nans"
        assert observations, observations

    return base_engine


@pytest.fixture
def seeded_session(p_seeded_engine):
    from sqlmodel import Session

    with Session(p_seeded_engine) as session:
        yield session


@pytest.fixture
def override_session(p_seeded_engine):
    def get_test_session():
        with Session(p_seeded_engine) as session:
            yield session

    app.dependency_overrides[get_session] = get_test_session
    yield
    app.dependency_overrides.clear()
