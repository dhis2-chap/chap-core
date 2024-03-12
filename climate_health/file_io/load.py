from .external_file import fetch_and_clean
from ..dataset import SpatioTemporalDataSet


def load_data_set(data_set_filename: str) -> SpatioTemporalDataSet:
    return fetch_and_clean(data_set_filename)