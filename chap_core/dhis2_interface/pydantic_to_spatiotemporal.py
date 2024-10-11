import pandas as pd

from chap_core.api_types import DataElement
from chap_core.datatypes import TimeSeriesArray
from chap_core.dhis2_interface.periods import convert_time_period_string
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


def v1_conversion(data_list: list[DataElement], fill_missing=False) -> DataSet[TimeSeriesArray]:
    """
    Convert a list of DataElement objects to a SpatioTemporalDict[TimeSeriesArray] object.
    """
    df = pd.DataFrame([d.dict() for d in data_list])
    df.sort_values(by=["ou", "pe"], inplace=True)
    d = dict(
        time_period=[convert_time_period_string(row) for row in df["pe"]],
        location=df.ou,
        value=df.value,
    )
    converted_df = pd.DataFrame(d)
    ds = DataSet.from_pandas(converted_df, TimeSeriesArray, fill_missing=fill_missing)
    return ds
