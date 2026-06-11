"""Tests for DataSetManager observation filtering and the flat-frame converter."""

import dataclasses
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel

from chap_core.database.dataset_manager import DataSetManager
from chap_core.database.dataset_tables import DataSetCreateInfo
from chap_core.spatio_temporal_data.converters import observations_to_dataframe

EXAMPLE_DATA = Path(__file__).parent.parent / "example_data"


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


def test_observations_period_range_no_match_returns_empty(manager, dataset_id):
    assert manager.observations(dataset_id, period_range=("1800-01", "1800-12")) == []


def test_observations_periods_non_contiguous_list(manager, dataset_id):
    periods = _sorted_periods(manager.observations(dataset_id))
    assert len(periods) >= 3
    wanted = [periods[0], periods[2]]  # deliberately skip periods[1] -> a gap
    filtered = manager.observations(dataset_id, periods=wanted)
    assert {obs.period for obs in filtered} == set(wanted)


def test_observations_period_range_composes_with_org_unit(manager, dataset_id):
    periods = _sorted_periods(manager.observations(dataset_id))
    assert len(periods) >= 4
    start, end = periods[1], periods[-2]
    org_unit = sorted({obs.org_unit for obs in manager.observations(dataset_id)})[0]
    filtered = manager.observations(dataset_id, period_range=(start, end), org_units=[org_unit])
    assert filtered
    assert all(obs.org_unit == org_unit and start <= obs.period <= end for obs in filtered)


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


def test_find_by_id_returns_row_or_none(manager, dataset_id):
    row = manager.find_by_id(dataset_id)
    assert row is not None and row.id == dataset_id
    assert manager.find_by_id(999999) is None


def test_find_all_and_delete_by_id(manager, dataset_id):
    assert [d.id for d in manager.find_all()] == [dataset_id]
    manager.delete_by_id(dataset_id)
    assert manager.find_all() == []
    assert manager.find_by_id(dataset_id) is None
    assert manager.observations(dataset_id) == []  # observations cascade-deleted with the dataset


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


def test_to_dataset_unknown_id_raises(manager):
    with pytest.raises(ValueError, match="not found"):
        manager.to_dataset(999999)


def test_find_by_name_returns_row_or_none(manager, dataset_id):
    row = manager.find_by_name("health_population")
    assert row is not None and row.id == dataset_id
    assert manager.find_by_name("does-not-exist") is None


def test_to_dataset_org_unit_filter_roundtrips(manager, dataset_id):
    org_unit = sorted({obs.org_unit for obs in manager.observations(dataset_id)})[0]
    dataset = manager.to_dataset(dataset_id, org_units=[org_unit])
    assert list(dataset.locations()) == [org_unit]


def test_to_dataset_period_range_windows_the_series(manager, dataset_id):
    all_periods = _sorted_periods(manager.observations(dataset_id))
    assert len(all_periods) >= 4
    start, end = all_periods[1], all_periods[-2]
    expected = [p for p in all_periods if start <= p <= end]
    windowed = manager.to_dataset(dataset_id, period_range=(start, end))
    full = manager.to_dataset(dataset_id)
    # the window drops the first and last period and keeps a consecutive run in between
    assert len(windowed.period_range) == len(expected)
    assert len(windowed.period_range) < len(full.period_range)


def test_to_dataset_period_range_and_org_unit_compose(manager, dataset_id):
    all_periods = _sorted_periods(manager.observations(dataset_id))
    assert len(all_periods) >= 4
    start, end = all_periods[1], all_periods[-2]
    org_unit = sorted({obs.org_unit for obs in manager.observations(dataset_id)})[0]
    dataset = manager.to_dataset(dataset_id, period_range=(start, end), org_units=[org_unit])
    assert list(dataset.locations()) == [org_unit]
    assert len(dataset.period_range) == len([p for p in all_periods if start <= p <= end])


def test_save_dataset_from_csv_with_geojson_loads_polygons(manager):
    dataset_id = manager.save_dataset_from_csv(
        "vietnam", EXAMPLE_DATA / "vietnam_monthly.csv", EXAMPLE_DATA / "vietnam_monthly.geojson"
    )
    dataset = manager.to_dataset(dataset_id)
    assert len(list(dataset.locations())) > 0
    assert dataset.polygons is not None
    assert manager.find_by_name("vietnam").period_type == "month"


def test_save_dataset_from_csv_weekly_sets_period_type(manager):
    manager.save_dataset_from_csv("nicaragua", EXAMPLE_DATA / "nicaragua_weekly_data.csv")
    assert manager.find_by_name("nicaragua").period_type == "week"
