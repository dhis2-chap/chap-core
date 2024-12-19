#from chap_core.main import assess_model_on_csv_data
import pytest

from chap_core.predictor.poisson import Poisson

from . import EXAMPLE_DATA_PATH

@pytest.mark.skip
def test_full_run():
    data_file = str(EXAMPLE_DATA_PATH / "monthly_data.csv")
    report = assess_model_on_csv_data(data_file, 0.5, Poisson())
    print(report.rmse_dict)
