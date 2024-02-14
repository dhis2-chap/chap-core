import os
from pathlib import Path
from typing import Annotated, Generator

from omnipy import StrDataset

from climate_health.data_wrangling.helpers import (load_data_as_clean_strings, standardize_separated_data,
                                                   )
import pytest

from pytest import fixture


EXAMPLE_DATA_PATH = Path(__file__).parent.parent.parent / 'example_data'

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



    # PerRegionRecordModel = create_pydantic_model_for_region_data('PerRegionRecordModel',
    #                                                              region_col_names=table_renamed_colnames_cleaned_ds.col_names[1:],
    #                                                              region_data_type=int | float)
    #
    # table_colnames_datatypes_ds = TableOfPydanticRecordsDataset[PerRegionRecordModel](table_renamed_colnames_cleaned_ds)