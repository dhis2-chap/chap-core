import pandas as pd
import pytest

from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


@pytest.mark.skip
def test_from_samples():
    assert False


def test_from_pandas():
    df = pd.DataFrame(
        {
            "location": ["Oslo", "Oslo", "Bergen", "Bergen"],
            "time_period": ["2020-01", "2020-02", "2020-01", "2020-02"],
            "disease_cases": [10, 20, 30, 40],
        }
    )
    ds = DataSet.from_pandas(df)
    print(ds)
