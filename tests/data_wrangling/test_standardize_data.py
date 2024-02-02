import os
from pathlib import Path
from typing import Annotated, Generator

from omnipy import StrDataset, SplitToLinesDataset, SplitLinesToColumnsDataset, PandasDataset

from climate_health.data_wrangling.helpers import (load_data_as_clean_strings, strip_commas,
                                                   rename_col_names,
                                                   create_pydantic_model_for_region_data)
import pytest

from pytest import fixture

from climate_health.data_wrangling.models import TableWithColNamesInFirstRowDataset, \
    TableWithColNamesDataset, TableOfPydanticRecordsDataset


@fixture(scope="module")
def separated_data() -> Annotated[Generator[StrDataset, None, None], pytest.fixture]:
    separate_data_path = Path(__file__).parent.parent.parent / 'example_data' / 'nonstandard_separate'
    ds = load_data_as_clean_strings(str(separate_data_path))
    yield ds
    os.unlink(f'{separate_data_path}.tar.gz')


@fixture(scope="module")
def separated_data_renamed(
        separated_data: Annotated[Generator[StrDataset, None, None], pytest.fixture]
) -> Annotated[StrDataset, pytest.fixture]:
    # separated_data.rename('separated_disease_data', 'disease')
    # separated_data.rename('separated_rain_data', 'rain')
    # separated_data.rename('separated_temp_data', 'temperature')

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
    lines_ds = SplitToLinesDataset(separated_data_renamed)
    items_ds = SplitLinesToColumnsDataset(lines_ds, delimiter=';')

    table_colnames_first_row_ds = TableWithColNamesInFirstRowDataset(items_ds)
    table_colnames_ds = TableWithColNamesDataset(table_colnames_first_row_ds)
    table_colnames_cleaned_ds = strip_commas.run(table_colnames_ds)
    table_renamed_colnames_cleaned_ds = rename_col_names.run(table_colnames_cleaned_ds, dict(periodname='time_period'))

    PerRegionRecordModel = create_pydantic_model_for_region_data('PerRegionRecordModel',
                                                                 region_col_names=table_renamed_colnames_cleaned_ds.col_names[1:],
                                                                 region_data_type=float)

    table_colnames_datatypes_ds = TableOfPydanticRecordsDataset[PerRegionRecordModel](table_renamed_colnames_cleaned_ds)
    pandas_ds = PandasDataset(table_colnames_datatypes_ds)
    print(pandas_ds['disease'])
    # tt = TableWithHeaderInFirstRowModel([['sdf', 'sd', 'd'], [1, 2, 3]])

#     standardized_data = join_and_pivot_separated_data(separated_data_renamed, delimiter=';',
#                                                       join_col='periodname', datafile_prefix='joined_')
#     assert isinstance(standardized_data, PandasDataset)
#     assert len(standardized_data) == 3
#     assert tuple(standardized_data.keys()) == ('joined_region1', 'joined_region2', 'joined_region3')
#     assert tuple(standardized_data['joined_region1'].columns) == ('disease', 'rain', 'temperature')
