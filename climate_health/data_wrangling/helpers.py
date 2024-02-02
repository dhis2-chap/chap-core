from omnipy import BytesDataset, StrDataset


def load_data_as_clean_strings(path_or_url: str) -> StrDataset:
    """
    Loads dataset as strings. Removes non-breaking space char in beginning of file, if present
    (see https://stackoverflow.com/a/50131187)
    """
    bytes_dataset = BytesDataset()
    bytes_dataset.load(path_or_url)
    return StrDataset(bytes_dataset, encoding='utf-8-sig')


def strip_commas(table_colnames_ds):
    for datafile in table_colnames_ds.keys():
        table_colnames_ds[datafile] = [
            {k: v.rstrip(',') if v is not None else None for k, v in tuple(row.items())}
            for row in table_colnames_ds[datafile]]
