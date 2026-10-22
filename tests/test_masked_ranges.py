import pytest
from pydantic import ValidationError

from chap_core.database.database import SessionWrapper
from chap_core.database.dataset_manager import DataSetManager
from chap_core.database.dataset_tables import DataConfiguration, DataSet, DataSetCreateInfo, MaskedRange


def test_masked_ranges_are_stored_and_read_back(health_population_data, engine):
    info = DataSetCreateInfo(name="masked", masked_ranges=[MaskedRange(start="2021-06", end="2021-08")])
    with SessionWrapper(engine) as session:
        dataset_id = DataSetManager(session.session).save_dataset(info, health_population_data, None)
        stored = session.session.get(DataSet, dataset_id)
        assert stored.masked_ranges == [MaskedRange(start="2021-06", end="2021-08")]


def test_dataset_without_masked_ranges_reads_back_empty(health_population_data, engine):
    with SessionWrapper(engine) as session:
        dataset_id = DataSetManager(session.session).save_dataset(
            DataSetCreateInfo(name="unmasked"), health_population_data, None
        )
        assert session.session.get(DataSet, dataset_id).masked_ranges == []


def test_create_info_accepts_camel_case_masked_ranges():
    info = DataSetCreateInfo.model_validate(
        {"name": "masked", "maskedRanges": [{"start": "2021-06", "end": "2021-08"}]}
    )
    assert info.masked_ranges == [MaskedRange(start="2021-06", end="2021-08")]


def test_data_configuration_parses_masked_ranges():
    config = DataConfiguration.model_validate({"maskedRanges": [{"start": "2021-06", "end": "2021-08"}]})
    assert config.masked_ranges == [MaskedRange(start="2021-06", end="2021-08")]


def test_data_configuration_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        DataConfiguration.model_validate({"maskedRanges": [], "columnMapping": {}})
