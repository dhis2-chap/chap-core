import dataclasses

import numpy as np

from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

def assert_tsdataclass_equal(ts_1, ts_2):
    """
    Assert that two time series data classes are equal.
    """
    assert len(ts_1) == len(ts_2)
    assert dataclasses.fields(ts_1) == dataclasses.fields(ts_2)
    for field in dataclasses.fields(ts_1):
        if field.name == "time_period":
            assert np.all(getattr(ts_1, field.name)==getattr(ts_2, field.name))
        else:
            np.testing.assert_allclose(getattr(ts_1, field.name), getattr(ts_2, field.name), rtol=1e-5, atol=1e-5)
            #np.testing.assert(getattr(ts_1, field.name), getattr(ts_2, field.name)), (field.name, getattr(ts_1, field.name), getattr(ts_2, field.name))



def assert_dataset_equal(dataset_1: DataSet, dataset_2: DataSet):
    """
    Assert that two datasets are equal.
    """
    assert set(dataset_1.keys()) == set(dataset_2.keys()), (set(dataset_1.keys()), set(dataset_2.keys()))
    for key in dataset_1.keys():
        assert_tsdataclass_equal(dataset_1[key], dataset_2[key])