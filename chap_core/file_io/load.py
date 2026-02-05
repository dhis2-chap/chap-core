from ..spatio_temporal_data.temporal_dataclass import DataSet
from .external_file import fetch_and_clean


def load_data_set(data_set_filename: str) -> DataSet:
    return fetch_and_clean(data_set_filename)  # type: ignore[no-any-return]
