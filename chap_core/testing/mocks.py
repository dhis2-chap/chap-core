from typing import Optional, List

import numpy as np

from chap_core.datatypes import SimpleClimateData, create_tsdataclass
from chap_core.rest_api_src.data_models import FetchRequest
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


class GEEMock:
    def __init__(self, *args, **kwargs):
        ...

    def get_historical_era5(self, features, periodes, fetch_requests: Optional[List[FetchRequest]] = None):
        if fetch_requests is None:
            feature_names = ['rainfall', 'mean_temperature']
        else:
            feature_names = [f.feature_name for f in fetch_requests]
        dataclass = create_tsdataclass(feature_names)
        locations = [f['id'] for f in features['features']]
        return DataSet({location:
                            dataclass(periodes, *[np.random.rand(len(periodes)) for _ in feature_names])
                        for location in locations})
