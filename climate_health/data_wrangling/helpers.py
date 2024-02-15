from omnipy import (BytesDataset,
                    StrDataset,
                    SplitToLinesDataset,
                    SplitLinesToColumnsDataset,
                    PandasDataset,
                    TableOfPydanticRecordsDataset,
                    TableWithColNamesDataset,
                    TableWithColNamesModel,
                    TaskTemplate,
                    rename_col_names,
                    transpose_columns_with_data_files)
from pydantic import create_model

from climate_health.datatypes import ClimateHealthTimeSeriesModel


def standardize_separated_data(separated_data: StrDataset) -> PandasDataset:
    lines_ds = SplitToLinesDataset(separated_data)
    items_ds = SplitLinesToColumnsDataset(lines_ds, delimiter=';')

    table_colnames_ds = TableWithColNamesDataset(items_ds)
    table_colnames_cleaned_ds = strip_commas.run(table_colnames_ds)

    table_transposed_ds = transpose_columns_with_data_files.run(table_colnames_cleaned_ds, exclude_cols=('periodname',))
    table_transposed_renamed_colnames_ds = rename_col_names.run(table_transposed_ds,
                                                             dict(periodname='time_period', rain='rainfall',
                                                                  temperature='mean_temperature',
                                                                  disease='disease_cases'))
    table_values_parsed_ds = TableOfPydanticRecordsDataset[ClimateHealthTimeSeriesModel](table_transposed_renamed_colnames_ds)
    return PandasDataset(table_values_parsed_ds)

def load_data_as_clean_strings(path_or_url: str) -> StrDataset:
    """
    Loads dataset as strings. Removes non-breaking space char in beginning of file, if present
    (see https://stackoverflow.com/a/50131187)
    """
    bytes_dataset = BytesDataset()
    bytes_dataset.load(path_or_url)
    return StrDataset(bytes_dataset, encoding='utf-8-sig')


@TaskTemplate(iterate_over_data_files=True, return_dataset_cls=TableWithColNamesDataset)
def strip_commas(data_file: TableWithColNamesModel) -> TableWithColNamesModel:
    # Possible alternative implementations based on future universal indexing functionality
    #
    # table_colnames_ds[:, :, 1:, :] = table_colnames_ds[:, :, 1:, :-1]
    # table_colnames_ds[:, :, 1:] = \
    #   table_colnames_ds[:, :, 1:].for_item(lambda k, v: (k, v.rstrip(',')))
    # table_colnames_ds[:, :, 1:] = table_colnames_ds[:, :, 1:].for_val(lambda v: v.rstrip(','))

    return TableWithColNamesModel([{k: v.rstrip(',') if v is not None else None
                                    for k, v in tuple(row.items())}
                                   for row in data_file])


def create_pydantic_model_for_region_data(model_name: str,
                                          region_col_names: tuple[str],
                                          region_data_type: type):
    fields = dict(time_period=(str, ...))
    for name in region_col_names:
        fields[name] = (region_data_type, ...)

    return create_model(model_name, **fields)

