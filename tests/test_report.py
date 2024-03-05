import pandas as pd
import pytest

from climate_health.reports import HTMLReport
from . import TMP_DATA_PATH


@pytest.fixture
def result_dict():
    return {'good_model': pd.DataFrame({'location': ['a', 'a', 'b'],
                                        'mae': [0.1, 0.2, 0.3],
                                        'period': ['2010-01-01', '2010-01-01', '2010-01-01']}),
            'bad_model': pd.DataFrame({'location': ['a', 'a', 'b'],
                                       'mae': [0.5, 0.3, 0.9],
                                       'period': ['2010-01-01', '2010-01-01', '2010-01-01']})}


@pytest.fixture
def tmp_path():
    path = TMP_DATA_PATH / 'output.md'
    assert not path.exists()
    return path

@pytest.mark.xfail
def test_from_results(result_dict, tmp_path):
    report = HTMLReport.from_results(result_dict)
    assert report is not None
    report.save(tmp_path)
    assert tmp_path.exists()
    # delete the file
    tmp_path.unlink()
