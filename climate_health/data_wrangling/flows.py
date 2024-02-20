from omnipy import FuncFlowTemplate, PandasDataset, SplitLinesToColumnsDataset, SplitToLinesDataset, StrDataset, \
    TableOfPydanticRecordsDataset, TableWithColNamesDataset, rename_col_names, transpose_columns_with_data_files

from climate_health.data_wrangling.tasks import strip_commas
from climate_health.datatypes import ClimateHealthTimeSeriesModel


@FuncFlowTemplate()
def standardize_separated_data(
        separated_data: StrDataset,
        time_period_col_name: str,
        rain_data_file_name: str,
        temp_data_file_name: str,
        disease_data_file_name: str
) -> PandasDataset:
    lines_ds = SplitToLinesDataset(separated_data)
    items_ds = SplitLinesToColumnsDataset(lines_ds, delimiter=';')

    table_colnames_ds = TableWithColNamesDataset(items_ds)
    table_colnames_cleaned_ds = strip_commas.run(table_colnames_ds)

    table_transposed_ds = transpose_columns_with_data_files.run(
        table_colnames_cleaned_ds,
        exclude_cols=(time_period_col_name,)
    )
    table_transposed_renamed_colnames_ds = rename_col_names.run(
        table_transposed_ds,
        prev2new_keymap={time_period_col_name: 'time_period',
                         temp_data_file_name: 'mean_temperature',
                         rain_data_file_name:'rainfall',
                         disease_data_file_name: 'disease_cases'}
    )
    table_values_parsed_ds = TableOfPydanticRecordsDataset[ClimateHealthTimeSeriesModel](table_transposed_renamed_colnames_ds)
    return PandasDataset(table_values_parsed_ds)
