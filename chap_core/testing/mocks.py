import numpy as np

from chap_core.datatypes import SimpleClimateData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


class GEEMock:
    def __init__(self, *args, **kwargs):
        ...

    def get_historical_era5(self, features, periodes):
        locations = [f['id'] for f in features['features']]
        return DataSet({location:
                            SimpleClimateData(periodes, np.random.rand(len(periodes)),
                                              np.random.rand(len(periodes)))
                        for location in locations})
