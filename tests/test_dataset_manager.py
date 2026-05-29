"""Tests for DataSetManager observation filtering and SessionWrapper delegation."""

import dataclasses

import pytest
from sqlalchemy import create_engine
from sqlmodel import SQLModel

from chap_core.database.database import SessionWrapper
from chap_core.database.dataset_tables import DataSetCreateInfo
from chap_core.datatypes import HealthPopulationData
from chap_core.spatio_temporal_data.converters import observations_to_dataframe


@pytest.fixture
def engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def dataset_id(engine, health_population_data):
    with SessionWrapper(engine) as session:
        return session.add_dataset(DataSetCreateInfo(name="health_population"), health_population_data, None)


def _sorted_periods(observations):
    return sorted({obs.period for obs in observations})


def test_get_observations_no_filter_returns_all(engine, dataset_id):
    with SessionWrapper(engine) as session:
        observations = session.datasets.get_observations(dataset_id)
    assert len(observations) > 0
    assert {obs.feature_name for obs in observations} == {"disease_cases", "population"}


def test_get_observations_filters_by_period_list(engine, dataset_id):
    with SessionWrapper(engine) as session:
        all_obs = session.datasets.get_observations(dataset_id)
        periods = _sorted_periods(all_obs)
        wanted = periods[:2]
        filtered = session.datasets.get_observations(dataset_id, periods=wanted)
    assert {obs.period for obs in filtered} == set(wanted)


def test_get_observations_filters_by_period_range(engine, dataset_id):
    with SessionWrapper(engine) as session:
        periods = _sorted_periods(session.datasets.get_observations(dataset_id))
        assert len(periods) >= 4
        start, end = periods[1], periods[-2]
        filtered = session.datasets.get_observations(dataset_id, period_range=(start, end))
    returned = {obs.period for obs in filtered}
    assert returned == {p for p in periods if start <= p <= end}
    assert periods[0] not in returned
    assert periods[-1] not in returned


def test_get_observations_filters_by_org_units(engine, dataset_id):
    with SessionWrapper(engine) as session:
        org_units = sorted({obs.org_unit for obs in session.datasets.get_observations(dataset_id)})
        wanted = org_units[:1]
        filtered = session.datasets.get_observations(dataset_id, org_units=wanted)
    assert {obs.org_unit for obs in filtered} == set(wanted)


def test_get_observations_filters_by_feature_names(engine, dataset_id):
    with SessionWrapper(engine) as session:
        filtered = session.datasets.get_observations(dataset_id, feature_names=["disease_cases"])
    assert {obs.feature_name for obs in filtered} == {"disease_cases"}


def test_get_observations_filters_compose(engine, dataset_id):
    with SessionWrapper(engine) as session:
        org_unit = sorted({obs.org_unit for obs in session.datasets.get_observations(dataset_id)})[0]
        filtered = session.datasets.get_observations(dataset_id, org_units=[org_unit], feature_names=["disease_cases"])
    assert all(obs.org_unit == org_unit and obs.feature_name == "disease_cases" for obs in filtered)
    assert len(filtered) > 0


def test_get_observations_empty_match_returns_empty(engine, dataset_id):
    with SessionWrapper(engine) as session:
        filtered = session.datasets.get_observations(dataset_id, periods=["1899-01"])
    assert filtered == []


def test_get_observations_unknown_dataset_raises(engine):
    with SessionWrapper(engine) as session:
        with pytest.raises(ValueError, match="not found"):
            session.datasets.get_observations(999999)


def test_observations_to_dataframe_shape(engine, dataset_id):
    with SessionWrapper(engine) as session:
        observations = session.datasets.get_observations(dataset_id, feature_names=["disease_cases"])
        df = observations_to_dataframe(observations)
    assert list(df.columns) == ["location", "time_period", "feature_name", "value"]
    assert (df["feature_name"] == "disease_cases").all()
    assert len(df) == len(observations)


def test_observations_to_dataframe_empty():
    df = observations_to_dataframe([])
    assert list(df.columns) == ["location", "time_period", "feature_name", "value"]
    assert df.empty


def test_get_dataset_feature_subset_narrows_dataclass(engine, dataset_id):
    with SessionWrapper(engine) as session:
        dataset = session.get_dataset(dataset_id, feature_names=["disease_cases"])
        _, data = next(iter(dataset.items()))
    field_names = {field.name for field in dataclasses.fields(data)}
    assert "disease_cases" in field_names
    assert "population" not in field_names


def test_get_dataset_empty_filter_raises(engine, dataset_id):
    with SessionWrapper(engine) as session:
        with pytest.raises(ValueError, match="No observations"):
            session.get_dataset(dataset_id, org_units=["does-not-exist"])


def test_get_dataset_org_unit_filter_roundtrips(engine, dataset_id):
    with SessionWrapper(engine) as session:
        org_unit = sorted({obs.org_unit for obs in session.datasets.get_observations(dataset_id)})[0]
        dataset = session.get_dataset(dataset_id, org_units=[org_unit])
        locations = list(dataset.locations())
    assert locations == [org_unit]
