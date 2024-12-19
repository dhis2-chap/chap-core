"""Console script for chap_core."""

import dataclasses
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from cyclopts import App

from chap_core.climate_predictor import QuickForecastFetcher
from chap_core.datatypes import FullData
from chap_core.external.external_model import get_model_maybe_yaml, get_model_from_directory_or_github_url
from chap_core.external.mlflow_wrappers import NoPredictionsError
from chap_core.log_config import initialize_logging
from chap_core.predictor.model_registry import naive_spec, registry
from chap_core.rest_api_src.v1.rest_api import get_openapi_schema
from chap_core.rest_api_src.worker_functions import samples_to_evaluation_response, dataset_to_datalist
from chap_core.spatio_temporal_data.multi_country_dataset import (
    MultiCountryDataSet,
)
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from . import api
from chap_core.dhis2_interface.ChapProgram import ChapPullPost
from chap_core.dhis2_interface.json_parsing import add_population_data

# from chap_core.external.models.jax_models.model_spec import (
#    SSMForecasterNuts,
#    NutsParams,
# )
# from chap_core.external.models.jax_models.specs import SSMWithoutWeather
from chap_core.plotting.prediction_plot import plot_forecast_from_summaries
from chap_core.predictor import ModelType, model_registry
from chap_core.file_io.example_data_set import datasets, DataSetType
from chap_core.time_period.date_util_wrapper import delta_month, Week
from .assessment.prediction_evaluator import evaluate_model, backtest as _backtest
from .assessment.forecast import multi_forecast as do_multi_forecast

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

app = App()


def append_to_csv(file_object, data_frame: pd.DataFrame):
    data_frame.to_csv(file_object, mode="a", header=False)


@app.command()
def evaluate(
        model_name: ModelType | str,
        dataset_name: DataSetType,
        dataset_country: Optional[str] = None,
        prediction_length: int = 6,
        n_splits: int = 7,
        report_filename: Optional[str] = "report.pdf",
        ignore_environment: bool = False,
        debug: bool = False,
        log_file: Optional[str] = None,
):
    """
    Evaluate a model on a dataset using forecast cross validation
    """
    initialize_logging(debug, log_file)
    logger.info(f"Evaluating model {model_name} on dataset {dataset_name}")
    
    dataset = datasets[dataset_name]
    dataset = dataset.load()

    if isinstance(dataset, MultiCountryDataSet):
        assert dataset_country is not None, "Must specify a country for multi country datasets"
        assert (
                dataset_country in dataset.countries
        ), f"Country {dataset_country} not found in dataset. Countries: {dataset.countries}"
        dataset = dataset[dataset_country]

    make_run_dir = debug
    model = get_model_from_directory_or_github_url(model_name, ignore_env=ignore_environment, make_run_dir=make_run_dir)
    model = model()
    try:
        results = evaluate_model(
            model,
            dataset,
            prediction_length=prediction_length,
            n_test_sets=n_splits,
            report_filename=report_filename,
        )
    except NoPredictionsError as e:
        logger.error(f"No predictions were made: {e}")
        return
    print(results)


@app.command()
def forecast(
        model_name: str,
        dataset_name: DataSetType,
        n_months: int,
        model_path: Optional[str] = None,
        out_path: Optional[str] = Path("./"),
):
    """
    Forecast n_months ahead using the given model and dataset

    Parameters:
        model_name: Name of the model to use, set to external to use an external model and specify the external model with model_path
        dataset_name: Name of the dataset to use, e.g. hydromet_5_filtered
        n_months: int: Number of months to forecast ahead
        model_path: Optional[str]: Path to the model if model_name is external. Can ge a github repo url starting with https://github.com and ending with .git or a path to a local directory.
        out_path: Optional[str]: Path to save the output file, default is the current directory
    """

    out_file = Path(out_path) / f"{model_name}_{dataset_name}_forecast_results_{n_months}.html"
    f = open(out_file, "w")
    figs = api.forecast(model_name, dataset_name, n_months, model_path)
    for fig in figs:
        f.write(fig.to_html())
    f.close()


@app.command()
def multi_forecast(
        model_name: str,
        dataset_name: DataSetType,
        n_months: int,
        pre_train_months: int,
        out_path: Path = Path(""),
):
    model, model_name = get_model_maybe_yaml(model_name)
    model = model()
    filename = out_path / f"{model_name}_{dataset_name}_multi_forecast_results_{n_months}.html"
    logger.info(f"Saving to {filename}")
    f = open(filename, "w")
    dataset = datasets[dataset_name].load()
    predictions_list = list(
        do_multi_forecast(
            model,
            dataset,
            n_months * delta_month,
            pre_train_delta=pre_train_months * delta_month,
        )
    )

    for location, true_data in dataset.items():
        local_predictions = [pred.get_location(location).data() for pred in predictions_list]
        fig = plot_forecast_from_summaries(local_predictions, true_data.data())
        f.write(fig.to_html())
    f.close()


@app.command()
def dhis_pull(base_url: str, username: str, password: str):
    path = Path("dhis2analyticsResponses/")
    path.mkdir(exist_ok=True, parents=True)
    from chap_core.dhis2_interface.ChapProgram import ChapPullPost

    process = ChapPullPost(
        dhis2Baseurl=base_url.rstrip("/"),
        dhis2Username=username,
        dhis2Password=password,
    )
    full_data_frame = get_full_dataframe(process)
    disease_filename = (path / process.DHIS2HealthPullConfig.get_id()).with_suffix(".csv")
    full_data_frame.to_csv(disease_filename)


def get_full_dataframe(process):
    disease_data_frame = process.pullDHIS2Analytics()
    population_data_frame = process.pullPopulationData()
    full_data_frame = add_population_data(disease_data_frame, population_data_frame)
    return full_data_frame


@app.command()
def dhis_flow(base_url: str, username: str, password: str, n_periods=1):
    process = ChapPullPost(
        dhis2Baseurl=base_url.rstrip("/"),
        dhis2Username=username,
        dhis2Password=password,
    )
    full_data_frame = get_full_dataframe(process)
    modelspec = naive_spec
    model = model_registry['naive_model']()
    model.train(full_data_frame)
    predictions = model.prediction_summary(Week(full_data_frame.end_timestamp))
    json_response = process.pushDataToDHIS2(predictions, modelspec.__class__.__name__, do_dict=False)
    # save json
    json_filename = Path("dhis2analyticsResponses/") / f"{modelspec.__class__.__name__}.json"
    with open(json_filename, "w") as f:
        json.dump(json_response, f, indent=4)


@app.command()
def serve(seedfile: Optional[str] = None, debug: bool = False):
    """
    Start CHAP as a backend server
    """
    from .rest_api_src.v1.rest_api import main_backend
    if seedfile is not None:
        data = json.load(open(seedfile))
    else:
        data = None
    main_backend(data)


@app.command()
def write_open_api_spec(out_path: str):
    """
    Write the OpenAPI spec to a file
    """
    schema = get_openapi_schema()
    with open(out_path, "w") as f:
        json.dump(schema, f, indent=4)


def base_args(func, *args, **kwargs):
    """
    Decorator that adds some base arguments to a command
    """
    base_args = [
        ("debug", bool, False),
        ("log_file", Optional[str], None)
    ]

    def new_func(*args, **kwargs):
        for arg_name, arg_type, default in base_args:
            if arg_name not in kwargs:
                kwargs[arg_name] = default
        return func(*args, **kwargs)


@app.command()
def test(**base_kwargs):
    """
    Simple test-command to check that the chap command works
    """
    initialize_logging()
   
    logger.debug("Debug message")
    logger.info("Info message")
    

@dataclasses.dataclass
class AreaPolygons: ...


"""@dataclasses.dataclass
class PredictionData:
    area_polygons: AreaPolygons
    health_data: SpatioTemporalDict[HealthData]
    climate_data: SpatioTemporalDict[ClimateData]
    population_data: SpatioTemporalDict[PopulationData]


def read_zip_folder(zip_file_path: str):
    #
    zip_file_reader = ZipFileReader(zip_file_path)
    ...


def convert_geo_json(geo_json_content) -> OurShapeFormat:
    ...

# def write_graph_data(geo_json_content) -> None:
#    ...





# GeoJson convertion
# zip folder reading
# GOthenburg
# Create prediction csv
"""


@app.command()
def dhis_zip_flow(
        zip_file_path: str,
        out_json: str,
        model_name: Optional[str] = None,
        docker_filename: Optional[str] = None,
):
    """
    Run an forecasting evaluation on  data from a zip file from DHIS2, and save the results to a json file
    Run using the specified model_name, which can also be a path to a yaml file. Optionally specify a docker filename

    Parameters:
        zip_file_path: str: Path to the zip file
        out_json: str: Path to the output json file
        model_name: Optional[str]: Name of the model to use, or path to a yaml file
        docker_filename: Optional[str]: Path to a docker file
    """

    api.dhis_zip_flow(zip_file_path, out_json, model_name, docker_filename=docker_filename)


@app.command()
def backtest(data_filename: Path, model_name: registry.model_type, out_folder: Path, prediction_length: int = 12, n_test_sets: int = 20, stride: int = 2):
    """
    Run a backtest on a dataset using the specified model

    Parameters:
        data_filename: Path: Path to the data file
        model_name: str: Name of the model to use
        out_folder: Path: Path to the output folder
    """
    dataset = DataSet.from_csv(data_filename, FullData)
    print(dataset)
    logger.info(f"Running backtest on {data_filename} with model {model_name}")
    logger.info(f"Dataset period range: {dataset.period_range}, locations: {list(dataset.locations())}")
    estimator = registry.get_model(model_name)
    predictions_list = _backtest(estimator, dataset, prediction_length=prediction_length,
                                 n_test_sets=n_test_sets, stride=stride, weather_provider=QuickForecastFetcher)
    response = samples_to_evaluation_response(
        predictions_list,
        quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
        real_data=dataset_to_datalist(dataset, 'dengue'))
    dataframe = pd.DataFrame([entry.dict() for entry in response.predictions])
    data_name = data_filename.stem
    dataframe.to_csv(out_folder / f'{data_name}_evaluation_{model_name}.csv')
    serialized_response = response.json()
    out_filename = out_folder / f'{data_name}_evaluation_response_{model_name}.json'

    with open(out_filename, 'w') as out_file:
        out_file.write(serialized_response)


def main_function():
    """
    This function should just be type hinted with common types,
    and it will run as a command line function
    Simple function<

    >>> main()

    """
    return



def main():
    app()


if __name__ == "__main__":
    app()
