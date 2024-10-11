from chap_core.omnipy_lib import (
    BytesDataset,
    StrDataset,
    TableWithColNamesDataset,
    TableWithColNamesModel,
    TaskTemplate,
)


@TaskTemplate()
def load_data_as_clean_strings(path_or_url: str) -> StrDataset:
    """
    Loads dataset as strings. Removes non-breaking space char in beginning of file, if present
    (see https://stackoverflow.com/a/50131187)
    """
    bytes_dataset = BytesDataset()
    bytes_dataset.load(path_or_url)
    return StrDataset(bytes_dataset, encoding="utf-8-sig")


@TaskTemplate(iterate_over_data_files=True, return_dataset_cls=TableWithColNamesDataset)
def strip_commas(data_file: TableWithColNamesModel) -> TableWithColNamesModel:
    # Possible alternative implementations based on future universal indexing functionality
    #
    # table_colnames_ds[:, :, 1:, :] = table_colnames_ds[:, :, 1:, :-1]
    # table_colnames_ds[:, :, 1:] = \
    #   table_colnames_ds[:, :, 1:].for_item(lambda k, v: (k, v.rstrip(',')))
    # table_colnames_ds[:, :, 1:] = table_colnames_ds[:, :, 1:].for_val(lambda v: v.rstrip(','))

    return TableWithColNamesModel(
        [{k: v.rstrip(",") if v is not None else None for k, v in tuple(row.items())} for row in data_file]
    )
