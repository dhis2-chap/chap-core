from climate_health.main import assess_model_on_csv_data
from climate_health.predictor.poisson import Poisson

from . import EXAMPLE_DATA_PATH


def test_full_run():
    data_file = str(EXAMPLE_DATA_PATH / "monthly_data.csv")
    report = assess_model_on_csv_data(data_file, 0.5, Poisson())
    print(report.rmse_dict)
