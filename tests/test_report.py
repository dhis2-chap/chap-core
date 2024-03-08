import pandas as pd
import pytest

from climate_health.reports import HTMLReport
from . import TMP_DATA_PATH


@pytest.fixture
def result_dict():
    return {'good_model': pd.DataFrame({'location': ['a', 'b', 'a', 'b', 'a', 'b'],
                                        'mae': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                                        'period': ['2010-01-01', '2010-01-01', '2011-05-01', '2011-05-01', '2012-09-01', '2012-09-01']}),
            'bad_model': pd.DataFrame({'location': ['a', 'b', 'a', 'b', 'a', 'b'],
                                       'mae': [0.5, 0.3, 0.9, 0.7, 0.1, 0.2],
                                       'period': ['2010-01-01', '2010-01-01', '2011-05-01', '2011-05-01', '2012-09-01', '2012-09-01']})}


@pytest.fixture
def tmp_path():
    path = TMP_DATA_PATH / 'output.html'
    assert not path.exists()
    return path


def test_from_results(result_dict, tmp_path):
    report = HTMLReport.from_results(result_dict)
    assert report is not None
    report.save(tmp_path)
    assert tmp_path.exists()
    # delete the file
    tmp_path.unlink()

