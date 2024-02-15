import os
import pytest

from typing import Annotated, Generator
from omnipy import StrDataset
from pytest import fixture

from climate_health.data_wrangling.helpers import (load_data_as_clean_strings, standardize_separated_data)

from .. import EXAMPLE_DATA_PATH


@fixture(scope="module")
def separated_data() -> Annotated[Generator[StrDataset, None, None], pytest.fixture]:
    separate_data_path = EXAMPLE_DATA_PATH / 'nonstandard_separate'

    ds = load_data_as_clean_strings(str(separate_data_path))
    yield ds
    os.unlink(f'{separate_data_path}.tar.gz')


@fixture(scope="module")
def separated_data_renamed(
        separated_data: Annotated[Generator[StrDataset, None, None], pytest.fixture]
) -> Annotated[StrDataset, pytest.fixture]:
    def rename(data, old_key: str, new_key: str):
        data[new_key] = data.pop(old_key)

    rename(separated_data, 'separated_disease_data', 'disease')
    rename(separated_data, 'separated_rain_data', 'rain')
    rename(separated_data, 'separated_temp_data', 'temperature')

    return separated_data


def test_load_separated_data(separated_data: Annotated[Generator[StrDataset, None, None], pytest.fixture]) -> None:
    assert isinstance(separated_data, StrDataset)
    assert len(separated_data) == 3
    assert tuple(separated_data.keys()) == ('separated_disease_data',
                                            'separated_rain_data',
                                            'separated_temp_data')
    assert separated_data['separated_disease_data'].startswith('periodname')


def test_standardize_separated_data(separated_data_renamed: Annotated[StrDataset, pytest.fixture]):
    standardized_table_ds = standardize_separated_data(separated_data_renamed)
    standardized_table_ds.save(str(EXAMPLE_DATA_PATH / 'nonstandard_separate_standardized'))
