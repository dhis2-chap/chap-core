from .external_file import fetch_and_clean
from ..spatio_temporal_data.temporal_dataclass import DataSet


def load_data_set(data_set_filename: str) -> DataSet:
    return fetch_and_clean(data_set_filename)
