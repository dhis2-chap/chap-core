import os

from climate_health.main import assess_model_on_csv_data, PlaceholderModel
from climate_health.predictor.poisson import Poisson

def get_file_name(filename):
    cwd = os.getcwd()
    if cwd.endswith("tests"):
        return os.path.join('../', filename)
    else:
        return filename
def test_full_run():
    data_file = get_file_name("example_data/data.csv")
    report = assess_model_on_csv_data(data_file, 0.5, Poisson())
    print(report.rmse_dict)
