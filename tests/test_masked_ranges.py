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


def test_data_configuration_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        DataConfiguration.model_validate({"maskedRanges": [], "columnMapping": {}})
