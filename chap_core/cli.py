"""Console script for chap_core."""

import dataclasses
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from cyclopts import App

from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.climate_predictor import QuickForecastFetcher
from chap_core.datatypes import FullData
from chap_core.external.external_model import get_model_maybe_yaml, get_model_from_directory_or_github_url
from chap_core.external.mlflow_wrappers import NoPredictionsError
from chap_core.log_config import initialize_logging
from chap_core.predictor.model_registry import registry

from chap_core.rest_api_src.worker_functions import samples_to_evaluation_response, dataset_to_datalist
from chap_core.spatio_temporal_data.multi_country_dataset import (
    MultiCountryDataSet,
)
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from . import api
from chap_core.plotting.prediction_plot import plot_forecast_from_summaries
from chap_core.predictor import ModelType
from chap_core.file_io.example_data_set import datasets, DataSetType
from chap_core.time_period.date_util_wrapper import delta_month
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
    if "," in model_name:
        # model_name is not only one model, but contains a list of models
        model_list = model_name.split(",")
    else:
        model_list = [model_name]
    
    results_dict = {}
    for name in model_list:
        model = get_model_from_directory_or_github_url(name, ignore_env=ignore_environment, make_run_dir=make_run_dir)
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
        results_dict[name] = results
    
    #need to iterate through the dict, like key and value or something and then extract the relevant metrics
    #to a pandas dataframe, and save it as a csv file.
    #it seems like results contain two dictionairies, one for aggregate metrics and one with seperate ones for each ts
    
    data = []
    first_model = True
    for key, value in results_dict.items():
        aggregate_metric_dist = value[0]
        row = [key]
        for k, v in aggregate_metric_dist.items():
            row.append(v)
        if first_model:
            data.append(["Model"] + list(aggregate_metric_dist.keys()))
            first_model = False
        data.append(row)
    dataframe = pd.DataFrame(data)
    csvname = Path(report_filename).with_suffix(".csv")

    # write dataframe to csvname
    dataframe.to_csv(csvname, index=False, header=False)
    logger.info(f"Evaluation complete. Results saved to {csvname}")



@app.command()
def sanity_check_model(model_url: str, use_local_environement: bool = False):
    '''
    Check that a model can be loaded, trained and used to make predictions
    '''
    dataset = datasets["hydromet_5_filtered"].load()
    train, tests = train_test_generator(dataset, 3, n_test_sets=2)
    context, future, truth = next(tests)
    try:
        model = get_model_from_directory_or_github_url(model_url, ignore_env=use_local_environement)
        estimator = model()
    except Exception as e:
        logger.error(f"Error while creating model: {e}")
        return False
    try:
        predictor = estimator.train(train)
    except Exception as e:
        logger.error(f"Error while training model: {e}")
        return False
    try:
        predictions = predictor.predict(context, future)
    except Exception as e:
        logger.error(f"Error while forecasting: {e}")
        return False
    for location, prediction in predictions.items():
        assert not np.isnan(prediction.samples).any(), f"NaNs in predictions for location {location}, {prediction.samples}"
    context, future, truth = next(tests)
    try:
        predictions = predictor.predict(context, future)
    except Exception as e:
        logger.error(f"Error while forecasting from a future time point: {e}")
        return False
    for location, prediction in predictions.items():
        assert not np.isnan(prediction.samples).any(), f"NaNs in futuresplit predictions for location {location}, {prediction.samples}"



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
def serve(seedfile: Optional[str] = None, debug: bool = False):
    """
    Start CHAP as a backend server
    """
    from .rest_api_src.v1.rest_api import main_backend
    logger.info("Running chap serve")
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
    from chap_core.rest_api_src.v1.rest_api import get_openapi_schema
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


@app.command()
def backtest(data_filename: Path, model_name: registry.model_type, out_folder: Path, prediction_length: int = 12,
             n_test_sets: int = 20, stride: int = 2):
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
    dataframe = pd.DataFrame([entry.model_dump() for entry in response.predictions])
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
