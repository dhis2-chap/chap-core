import logging
from pathlib import Path
logger = logging.getLogger(__name__)
import json
import pandas as pd
import pytest
from chap_core.api_types import PredictionRequest
from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.datatypes import FullData
from chap_core.exceptions import InvalidDateError
from chap_core.models.utils import get_model_from_directory_or_github_url
from chap_core.geometry import Polygons
from chap_core.rest_api_src.worker_functions import dataset_from_request_v1
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period.date_util_wrapper import TimePeriod
from tests.conftest import data_path
from tests.integration.rest_api.test_integration import run_job_and_get_status


# hacky stuff to try to make a compatible csv file for now:
"""
data = pd.read_csv(data_path / "small_laos_data_with_polygons.csv")
# make time_period column from year and week
# translate year and week to string "YYYYwWW"
data["time_period"] = data["year"].astype(str) + "W" + data["week"].astype(str).str.zfill(2)

# find rows with invalid time period
time_periods = data["time_period"]
valids = []
for t in time_periods:
    try:
        t = TimePeriod.parse(t)
    except InvalidDateError:
        print("INVALID DATE:", t)
        valids.append(False)
    else:
        valids.append(True)
data = data[valids]

# write to csv
data.to_csv(data_path / "small_laos_data_with_polygons2.csv", index=False)
"""

#data = DataSet.from_csv(data_path / "small_laos_data_with_polygons2.csv", FullData)
#data = DataSet.from_csv("/home/ivargry/dev/ewars_plus_python_wrapper/demo_data/laos_dengue_and_diarrhea_surv_data_2015_2023_chap_format.csv", FullData)

data_path = Path("example_data/")

def add_time_periods_to_data(data):
    data["time_period"] = data["year"].astype(str) + "W" + data["week"].astype(str).str.zfill(2)
    
    # find rows with invalid time period
    time_periods = data["time_period"]
    valids = []
    for t in time_periods:
        try:
            t = TimePeriod.parse(t)
        except InvalidDateError:
            print("INVALID DATE:", t)
            valids.append(False)
        else:
            valids.append(True)
    data = data[valids]
    return data


province_name_to_keep = ["01 Vientiane Capital", "09 Xiangkhouang"]
data = pd.read_csv("/home/ivargry/dev/ewars_plus_python_wrapper/demo_data/laos_dengue_and_diarrhea_surv_data_2015_2023_chap_format.csv")
#data = data[data["province_name"].isin(province_name_to_keep)]
data = add_time_periods_to_data(data)
# change district column to string
print(data)
data["location"] = data["location"].astype(str)
logger.error(data["time_period"])
data = DataSet.from_pandas(data, FullData)
logger.error(data.to_pandas()["time_period"])
output_dir = Path("target")
output_dir.mkdir(exist_ok=True)
test_csv_path = output_dir / "test.csv"
data.to_csv(str(test_csv_path))
exit()
print("---- Data before  ----")
print(data)
data = DataSet.from_csv(str(test_csv_path), FullData)
print("---- Data after ----")
print(data)
 
polygons = Polygons.from_file(data_path / "small_laos_data_with_polygons.geojson", id_property="district")
polygons.filter_locations(data.locations())
polygons = polygons.data

#polygons = Polygons.from_file("/home/ivargry/dev/ewars_plus_python_wrapper/demo_data/laos_province_shapefile.GEOJSON").data
data.set_polygons(polygons)

external_model = get_model_from_directory_or_github_url(
    #"https://github.com/dhis2-chap/ewars_plus_python_wrapper",
    "/home/ivargry/dev/ewars_plus_python_wrapper",
    ignore_env=True,
    run_dir_type="latest"
)
#external_model.train(data) 
evaluate_model(external_model, data, report_filename='test_integration_report.pdf', n_test_sets=1)
