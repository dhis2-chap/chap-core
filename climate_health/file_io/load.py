from .external_file import fetch_and_clean
from .._legacy_dataset import IsSpatioTemporalDataSet


def load_data_set(data_set_filename: str) -> IsSpatioTemporalDataSet:
    return fetch_and_clean(data_set_filename)
