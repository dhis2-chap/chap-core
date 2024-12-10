from omnipy import (
    DagFlowTemplate,
    FuncFlowTemplate,
    LinearFlowTemplate,
    PandasDataset,
    SplitLinesToColumnsDataset,
    SplitToLinesDataset,
    StrDataset,
    TableOfPydanticRecordsDataset,
    TableWithColNamesDataset,
    convert_dataset,
    rename_col_names,
    transpose_columns_with_data_files,
)

from chap_core.data_wrangling.tasks import strip_commas
from chap_core.datatypes import ClimateHealthTimeSeriesModel


def create_prev2new_keymap(
    time_period_col_name: str,
    rain_data_file_name: str,
    temp_data_file_name: str,
    disease_data_file_name: str,
) -> dict[str, str]:
    return {
        time_period_col_name: "time_period",
        rain_data_file_name: "rainfall",
        temp_data_file_name: "mean_temperature",
        disease_data_file_name: "disease_cases",
    }


@FuncFlowTemplate()
def standardize_separated_data_func_flow(
    separated_data: StrDataset,
    time_period_col_name: str,
    rain_data_file_name: str,
    temp_data_file_name: str,
    disease_data_file_name: str,
) -> PandasDataset:
    lines_ds = SplitToLinesDataset(separated_data)
    items_ds = SplitLinesToColumnsDataset(lines_ds, delimiter=";")

    table_colnames_ds = TableWithColNamesDataset(items_ds)
    table_colnames_cleaned_ds = strip_commas.run(table_colnames_ds)

    table_transposed_ds = transpose_columns_with_data_files.run(
        table_colnames_cleaned_ds, exclude_cols=(time_period_col_name,)
    )
    table_transposed_renamed_colnames_ds = rename_col_names.run(
        table_transposed_ds,
        prev2new_keymap=create_prev2new_keymap(
            time_period_col_name,
            rain_data_file_name,
            temp_data_file_name,
            disease_data_file_name,
        ),
    )
    table_values_parsed_ds = TableOfPydanticRecordsDataset[ClimateHealthTimeSeriesModel](
        table_transposed_renamed_colnames_ds
    )
    return PandasDataset(table_values_parsed_ds)


@LinearFlowTemplate(
    convert_dataset.refine(fixed_params=dict(dataset_cls=SplitToLinesDataset)),
    convert_dataset.refine(fixed_params=dict(dataset_cls=SplitLinesToColumnsDataset, delimiter=";")),
    convert_dataset.refine(fixed_params=dict(dataset_cls=TableWithColNamesDataset)),
    strip_commas,
    transpose_columns_with_data_files,
    rename_col_names,
    convert_dataset.refine(fixed_params=dict(dataset_cls=TableOfPydanticRecordsDataset[ClimateHealthTimeSeriesModel])),
    convert_dataset.refine(fixed_params=dict(dataset_cls=PandasDataset)),
)
def standardize_separated_data_linear_flow(
    dataset: StrDataset,
    exclude_cols: tuple[str, ...],
    prev2new_keymap: dict[str, str],
) -> PandasDataset: ...


@DagFlowTemplate(
    convert_dataset.refine(fixed_params=dict(dataset_cls=SplitToLinesDataset)),
    convert_dataset.refine(fixed_params=dict(dataset_cls=SplitLinesToColumnsDataset, delimiter=";")),
    convert_dataset.refine(fixed_params=dict(dataset_cls=TableWithColNamesDataset)),
    strip_commas,
    transpose_columns_with_data_files,
    rename_col_names,
    convert_dataset.refine(fixed_params=dict(dataset_cls=TableOfPydanticRecordsDataset[ClimateHealthTimeSeriesModel])),
    convert_dataset.refine(fixed_params=dict(dataset_cls=PandasDataset)),
    for_all_subjobs=dict(ensure_result_keys=["dataset"]),
    expand_result_key="dataset",
)
def standardize_separated_data_dag_flow(
    dataset: StrDataset,
    exclude_cols: tuple[str, ...],
    prev2new_keymap: dict[str, str],
) -> PandasDataset: ...


@FuncFlowTemplate()
def standardize_separated_data_wrapper(
    standardize_separated_data_flow_template: type[LinearFlowTemplate, DagFlowTemplate],
    separated_data: StrDataset,
    time_period_col_name: str,
    rain_data_file_name: str,
    temp_data_file_name: str,
    disease_data_file_name: str,
):
    return standardize_separated_data_flow_template(
        separated_data,
        exclude_cols=(time_period_col_name,),
        prev2new_keymap=create_prev2new_keymap(
            time_period_col_name,
            rain_data_file_name,
            temp_data_file_name,
            disease_data_file_name,
        ),
    )
