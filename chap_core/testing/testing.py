from bionumpy.util.testing import assert_bnpdataclass_equal

from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


def assert_dataset_equal(dataset_1: DataSet, dataset_2: DataSet):
    """
    Assert that two datasets are equal.
    """
    assert set(dataset_1.keys()) == set(dataset_2.keys()), (set(dataset_1.keys()), set(dataset_2.keys()))
    for key in dataset_1.keys():
        assert_bnpdataclass_equal(dataset_1[key], dataset_2[key])