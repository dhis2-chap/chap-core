import numpy as np
from bionumpy import replace

from climate_health.datatypes import TimeSeriesData
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict
from climate_health.time_period.date_util_wrapper import TimeStamp


def mask_covid_data(data: SpatioTemporalDict, start_date: TimeStamp = TimeStamp.parse('2020-03'),
                    end_date: TimeStamp=TimeStamp.parse('2021-12-31')) -> SpatioTemporalDict:
    """
    Mask the covid data to the specified date range
    """
    def insert_nans(ts: TimeSeriesData):
        mask_1 = (ts.time_period >= start_date)
        mask_2 = (ts.time_period <= end_date)
        mask = mask_1 & mask_2
        disease_cases = np.where(~mask, ts.disease_cases, np.nan)
        return replace(ts, disease_cases=disease_cases)
    new_dict = SpatioTemporalDict({location: insert_nans(data) for location, data in data.items()})
    return new_dict