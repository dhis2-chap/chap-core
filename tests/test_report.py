import pandas as pd
import pytest

from chap_core._legacy.reports import HTMLReport


@pytest.fixture
def result_dict():
    return {
        "good_model": pd.DataFrame(
            {
                "location": ["a", "b", "a", "b", "a", "b"],
                "mae": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "mle": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "period": [
                    "2010-01-01",
                    "2010-01-01",
                    "2011-05-01",
                    "2011-05-01",
                    "2012-09-01",
                    "2012-09-01",
                ],
            }
        ),
        "bad_model": pd.DataFrame(
            {
                "location": ["a", "b", "a", "b", "a", "b"],
                "mae": [0.5, 0.3, 0.9, 0.7, 0.1, 0.2],
                "mle": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "period": [
                    "2010-01-01",
                    "2010-01-01",
                    "2011-05-01",
                    "2011-05-01",
                    "2012-09-01",
                    "2012-09-01",
                ],
            }
        ),
    }


@pytest.fixture
def result_path(tmp_path):
    path = tmp_path / "output.html"
    assert not path.exists()
    return path


def test_from_results(result_dict, result_path):
    report = HTMLReport.from_results(result_dict)
    assert report is not None
    report.save(result_path)
    assert result_path.exists()
    # delete the file
    # result.unlink()
