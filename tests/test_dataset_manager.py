"""Tests for DataSetManager observation filtering and the flat-frame converter."""

import dataclasses

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel

from chap_core.database.dataset_manager import DataSetManager
from chap_core.database.dataset_tables import DataSetCreateInfo
from chap_core.spatio_temporal_data.converters import observations_to_dataframe


@pytest.fixture
def engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def dataset_id(engine, health_population_data):
    with Session(engine) as session:
        return DataSetManager(session).save_dataset(
            DataSetCreateInfo(name="health_population"), health_population_data, None
        )


@pytest.fixture
def manager(engine):
    with Session(engine) as session:
        yield DataSetManager(session)


def _sorted_periods(observations):
    return sorted({obs.period for obs in observations})


def test_observations_no_filter_returns_all(manager, dataset_id):
    observations = manager.observations(dataset_id)
    assert len(observations) > 0
    assert {obs.feature_name for obs in observations} == {"disease_cases", "population"}


def test_observations_filters_by_period_list(manager, dataset_id):
    periods = _sorted_periods(manager.observations(dataset_id))
    wanted = periods[:2]
    filtered = manager.observations(dataset_id, periods=wanted)
    assert {obs.period for obs in filtered} == set(wanted)


def test_observations_filters_by_period_range(manager, dataset_id):
    periods = _sorted_periods(manager.observations(dataset_id))
    assert len(periods) >= 4
    start, end = periods[1], periods[-2]
    filtered = manager.observations(dataset_id, period_range=(start, end))
    returned = {obs.period for obs in filtered}
    assert returned == {p for p in periods if start <= p <= end}
    assert periods[0] not in returned
    assert periods[-1] not in returned


def test_observations_filters_by_org_units(manager, dataset_id):
    org_units = sorted({obs.org_unit for obs in manager.observations(dataset_id)})
    wanted = org_units[:1]
    filtered = manager.observations(dataset_id, org_units=wanted)
    assert {obs.org_unit for obs in filtered} == set(wanted)


def test_observations_filters_by_feature_names(manager, dataset_id):
    filtered = manager.observations(dataset_id, feature_names=["disease_cases"])
    assert {obs.feature_name for obs in filtered} == {"disease_cases"}


def test_observations_filters_compose(manager, dataset_id):
    org_unit = sorted({obs.org_unit for obs in manager.observations(dataset_id)})[0]
    filtered = manager.observations(dataset_id, org_units=[org_unit], feature_names=["disease_cases"])
    assert all(obs.org_unit == org_unit and obs.feature_name == "disease_cases" for obs in filtered)
    assert len(filtered) > 0


def test_observations_empty_match_returns_empty(manager, dataset_id):
    assert manager.observations(dataset_id, periods=["1899-01"]) == []


def test_observations_unknown_dataset_returns_empty(manager):
    assert manager.observations(999999) == []


def test_observations_to_dataframe_shape(manager, dataset_id):
    observations = manager.observations(dataset_id, feature_names=["disease_cases"])
    df = observations_to_dataframe(observations)
    assert list(df.columns) == ["location", "time_period", "feature_name", "value"]
    assert (df["feature_name"] == "disease_cases").all()
    assert len(df) == len(observations)


def test_observations_to_dataframe_empty():
    df = observations_to_dataframe([])
    assert list(df.columns) == ["location", "time_period", "feature_name", "value"]
    assert df.empty


def test_to_dataset_feature_subset_narrows_dataclass(manager, dataset_id):
    dataset = manager.to_dataset(dataset_id, feature_names=["disease_cases"])
    _, data = next(iter(dataset.items()))
    field_names = {field.name for field in dataclasses.fields(data)}
    assert "disease_cases" in field_names
    assert "population" not in field_names


def test_to_dataset_empty_filter_raises(manager, dataset_id):
    with pytest.raises(ValueError, match="No observations"):
        manager.to_dataset(dataset_id, org_units=["does-not-exist"])


def test_to_dataset_org_unit_filter_roundtrips(manager, dataset_id):
    org_unit = sorted({obs.org_unit for obs in manager.observations(dataset_id)})[0]
    dataset = manager.to_dataset(dataset_id, org_units=[org_unit])
    assert list(dataset.locations()) == [org_unit]
