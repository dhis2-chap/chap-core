import numpy as np
import pytest
from pydantic import ValidationError

from chap_core.database.database import SessionWrapper
from chap_core.database.dataset_manager import DataSetManager
from chap_core.database.dataset_tables import DataConfiguration, DataSet, DataSetCreateInfo, MaskedRange


def test_masked_ranges_are_stored_and_read_back(health_population_data, engine):
    info = DataSetCreateInfo(name="masked", masked_ranges=[MaskedRange(start="2021W1", end="2021W3")])
    with SessionWrapper(engine) as session:
        dataset_id = DataSetManager(session.session).save_dataset(info, health_population_data, None)
        stored = session.session.get(DataSet, dataset_id)
        assert stored.masked_ranges == [MaskedRange(start="2021W01", end="2021W03")]


def test_masked_range_name_is_optional_and_stored(health_population_data, engine):
    ranges = [MaskedRange(start="2021W1", end="2021W3", name="outage"), MaskedRange(start="2021W10", end="2021W12")]
    with SessionWrapper(engine) as session:
        dataset_id = DataSetManager(session.session).save_dataset(
            DataSetCreateInfo(name="named", masked_ranges=ranges), health_population_data, None
        )
        stored = session.session.get(DataSet, dataset_id).masked_ranges
        assert stored is not None
        assert [(r.name, r.start) for r in stored] == [("outage", "2021W01"), (None, "2021W10")]


def test_dataset_without_masked_ranges_reads_back_empty(health_population_data, engine):
    with SessionWrapper(engine) as session:
        dataset_id = DataSetManager(session.session).save_dataset(
            DataSetCreateInfo(name="unmasked"), health_population_data, None
        )
        assert session.session.get(DataSet, dataset_id).masked_ranges == []


def test_month_ranges_are_rejected_on_a_weekly_dataset(health_population_data, engine):
    info = DataSetCreateInfo(name="mismatch", masked_ranges=[MaskedRange(start="2021-06", end="2021-08")])
    with SessionWrapper(engine) as session, pytest.raises(ValueError, match="month range"):
        DataSetManager(session.session).save_dataset(info, health_population_data, None)


@pytest.mark.parametrize(
    "start, end, expected",
    [
        ("2021-6", "2021-08", ("202106", "202108")),
        ("202106", "202108", ("202106", "202108")),
        (202106, 202108, ("202106", "202108")),
        ("2021W1", "2021-W3", ("2021W01", "2021W03")),
        ("2021-S01", "2021SunW2", ("2021SunW01", "2021SunW02")),
    ],
)
def test_masked_range_bounds_are_normalised(start, end, expected):
    masked_range = MaskedRange(start=start, end=end)
    assert (masked_range.start, masked_range.end) == expected


@pytest.mark.parametrize(
    "start, end",
    [
        ("banana", "2021-02"),
        ("2021-08", "2021-06"),
        ("2021W1", "2021-03"),
        ("2021W01", "2021SunW02"),
        ("2021", "2021"),
        ("2021-07-01", "2021-07-02"),
        ("2021W53", "2021W53"),
    ],
)
def test_invalid_masked_range_is_rejected(start, end):
    with pytest.raises(ValidationError):
        MaskedRange(start=start, end=end)


@pytest.mark.parametrize(
    "ranges",
    [
        [("2021-01", "2021-03"), ("2021-03", "2021-05")],
        [("2021-01", "2021-03"), ("2021W10", "2021W12")],
    ],
)
def test_overlapping_or_mixed_ranges_are_rejected(ranges):
    with pytest.raises(ValidationError):
        DataSetCreateInfo(name="x", masked_ranges=[MaskedRange(start=start, end=end) for start, end in ranges])


def test_create_info_accepts_camel_case_masked_ranges():
    info = DataSetCreateInfo.model_validate(
        {"name": "masked", "maskedRanges": [{"start": "2021-06", "end": "2021-08"}]}
    )
    assert info.masked_ranges == [MaskedRange(start="202106", end="202108")]


def test_data_configuration_parses_masked_ranges():
    config = DataConfiguration.model_validate({"maskedRanges": [{"start": "2021-06", "end": "2021-08"}]})
    assert config.masked_ranges == [MaskedRange(start="202106", end="202108")]


def test_disease_cases_are_masked_when_the_dataset_is_saved(health_population_data, engine):
    """By design, the stored observations are masked: NaN inside the ranges, unchanged elsewhere."""
    periods = health_population_data.period_range
    info = DataSetCreateInfo(name="masked", masked_ranges=[MaskedRange(start=periods[10].id, end=periods[12].id)])
    with SessionWrapper(engine) as session:
        manager = DataSetManager(session.session)
        dataset_id = manager.save_dataset(info, health_population_data, None)
        stored = manager.to_dataset(dataset_id)
        assert "target_masked" not in stored.field_names()
        for location, data in stored.items():
            original = health_population_data[location].disease_cases
            assert np.all(np.isnan(data.disease_cases[10:13]))
            assert np.array_equal(data.disease_cases[:10], original[:10], equal_nan=True)
            assert np.array_equal(data.disease_cases[13:], original[13:], equal_nan=True)


def test_saving_does_not_change_the_input_dataset(health_population_data, engine):
    periods = health_population_data.period_range
    location = next(iter(health_population_data.keys()))
    before = health_population_data[location].disease_cases.copy()
    info = DataSetCreateInfo(name="masked", masked_ranges=[MaskedRange(start=periods[10].id, end=periods[12].id)])
    with SessionWrapper(engine) as session:
        DataSetManager(session.session).save_dataset(info, health_population_data, None)
    assert np.array_equal(health_population_data[location].disease_cases, before, equal_nan=True)
