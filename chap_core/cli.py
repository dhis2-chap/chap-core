"""Console script for chap_core."""

import logging
import dataclasses
import json
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import yaml
from cyclopts import App

from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.assessment.forecast import multi_forecast as do_multi_forecast
from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.datatypes import FullData
from chap_core.exceptions import NoPredictionsError
from chap_core.hpo.searcher import RandomSearcher
from chap_core.models.model_template import ModelTemplate
from chap_core.models.utils import (
    get_model_from_directory_or_github_url,
    get_model_template_from_directory_or_github_url,
)
from chap_core.geometry import Polygons
from chap_core.log_config import initialize_logging
from chap_core.plotting.dataset_plot import StandardizedFeaturePlot
from chap_core.plotting.prediction_plot import plot_forecast_from_summaries
from chap_core.plotting.season_plot import SeasonCorrelationBarPlot
from chap_core.predictor import ModelType
from chap_core.spatio_temporal_data.multi_country_dataset import (
    MultiCountryDataSet,
)
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core import api
from chap_core.file_io.example_data_set import datasets, DataSetType
from chap_core.time_period.date_util_wrapper import delta_month

from chap_core.hpo.hpoModel import HpoModel, Direction
from chap_core.hpo.objective import Objective 
from chap_core.hpo.base import load_search_space_from_yaml


logger = logging.getLogger()
logger.setLevel(logging.INFO)

app = App()


def append_to_csv(file_object, data_frame: pd.DataFrame):
    data_frame.to_csv(file_object, mode="a", header=False)


@app.command()
def evaluate_hpo(
    model_name: ModelType | str,
    dataset_name: Optional[DataSetType] = None,
    dataset_country: Optional[str] = None,
    dataset_csv: Optional[Path] = None,
    polygons_json: Optional[Path] = None,
    polygons_id_field: Optional[str] = "id",
    prediction_length: int = 3,
    n_splits: int = 7,
    report_filename: Optional[str] = "report.pdf",
    ignore_environment: bool = False,
    debug: bool = False,
    log_file: Optional[str] = None,
    run_directory_type: Optional[Literal["latest", "timestamp", "use_existing"]] = "timestamp",
    model_configuration_yaml: Optional[str] = None,
    metric: Optional[str] = "MSE",
    direction: Direction = "minimize",
    evaluate_hpo: Optional[bool] = True,
):
    """
    Same as evaluate, but has three added arguments and a if check on argument evaluate_hpo. 
    """
    initialize_logging(debug, log_file)
    if dataset_name is None:
        assert dataset_csv is not None, "Must specify a dataset name or a dataset csv file"
        logging.info(f"Loading dataset from {dataset_csv}")
        dataset = DataSet.from_csv(dataset_csv, FullData)
        if polygons_json is not None:
            logging.info(f"Loading polygons from {polygons_json}")
            polygons = Polygons.from_file(polygons_json, id_property=polygons_id_field)
            polygons.filter_locations(dataset.locations())
            dataset.set_polygons(polygons.data)
    else:
        logger.info(f"Evaluating model {model_name} on dataset {dataset_name}")

        dataset = datasets[dataset_name]
        dataset = dataset.load()

        if isinstance(dataset, MultiCountryDataSet):
            assert dataset_country is not None, "Must specify a country for multi country datasets"
            assert (
                dataset_country in dataset.countries
            ), f"Country {dataset_country} not found in dataset. Countries: {dataset.countries}"
            dataset = dataset[dataset_country]

    if "," in model_name:
        # model_name is not only one model, but contains a list of models
        model_list = model_name.split(",")
        model_configuration_yaml_list = [None for _ in model_list]
        if model_configuration_yaml is not None:
            model_configuration_yaml_list = model_configuration_yaml.split(",")
            assert len(model_list) == len(
                model_configuration_yaml_list
            ), "Number of model configurations does not match number of models"
    else:
        model_list = [model_name]
        model_configuration_yaml_list = [model_configuration_yaml]

    logging.info(f"Model configuration: {model_configuration_yaml_list}")

    results_dict = {}
    for name, configuration in zip(model_list, model_configuration_yaml_list):
        if not evaluate_hpo:
            template = ModelTemplate.from_directory_or_github_url(
                name,
                base_working_dir=Path("./runs/"),
                ignore_env=ignore_environment,
                run_dir_type=run_directory_type,
            )
            logging.info(f"Model template loaded: {template}")
            if configuration is not None:
                logger.info(f"Loading model configuration from yaml file {configuration}")
                configuration = ModelConfiguration.model_validate(
                    yaml.safe_load(open(configuration))
                )  # template.get_model_configuration_from_yaml(Path(configuration))
                logger.info(f"Loaded model configuration from yaml file: {configuration}")

            model = template.get_model(configuration)
            model = model()
        else:
            if configuration is not None:
                logger.info(f"Loading model configuration from yaml file {configuration}")
                # base_configs = ModelConfiguration.model_validate(
                #     yaml.safe_load(open(configuration))
                # )
                # with open(configuration, "r", encoding="utf-8") as f:
                #     base_configs = yaml.safe_load(f) or {} # check if this returns a dict
                configs = load_search_space_from_yaml(configuration)
                # base_configs = {"user_option_values": configs}
                logger.info(f"Loaded model base configurations from yaml file: {configs}")

            # if "user_option_values" not in base_configs or not isinstance(base_configs["user_option_values"], dict):
            #     raise ValueError("Expected top-level key 'user_option_values' mapping to a dict of lists.")
            
            print("Creating HpoModel")
            objective = Objective(name, metric, prediction_length, n_splits)
            model = HpoModel(RandomSearcher(2), objective, direction, configs)
        try:
            results = evaluate_model(
                estimator=model,
                data=dataset,
                prediction_length=prediction_length,
                n_test_sets=n_splits,
                report_filename=report_filename,
            )
        except NoPredictionsError as e:
            logger.error(f"No predictions were made: {e}")
            return
        print(f"Results: {results}")
        results_dict[name] = results

    # need to iterate through the dict, like key and value or something and then extract the relevant metrics
    # to a pandas dataframe, and save it as a csv file.
    # it seems like results contain two dictionairies, one for aggregate metrics and one with seperate ones for each ts

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

    return results_dict


@app.command()
def evaluate(
    model_name: ModelType | str,
    dataset_name: Optional[DataSetType] = None,
    dataset_country: Optional[str] = None,
    dataset_csv: Optional[Path] = None,
    polygons_json: Optional[Path] = None,
    polygons_id_field: Optional[str] = "id",
    prediction_length: int = 6,
    n_splits: int = 7,
    report_filename: Optional[str] = "report.pdf",
    ignore_environment: bool = False,
    debug: bool = False,
    log_file: Optional[str] = None,
    run_directory_type: Optional[Literal["latest", "timestamp", "use_existing"]] = "timestamp",
    model_configuration_yaml: Optional[str] = None,
):
    initialize_logging(debug, log_file)
    if dataset_name is None:
        assert dataset_csv is not None, "Must specify a dataset name or a dataset csv file"
        logging.info(f"Loading dataset from {dataset_csv}")
        dataset = DataSet.from_csv(dataset_csv, FullData)
        if polygons_json is not None:
            logging.info(f"Loading polygons from {polygons_json}")
            polygons = Polygons.from_file(polygons_json, id_property=polygons_id_field)
            polygons.filter_locations(dataset.locations())
            dataset.set_polygons(polygons.data)
    else:
        logger.info(f"Evaluating model {model_name} on dataset {dataset_name}")

        dataset = datasets[dataset_name]
        dataset = dataset.load()

        if isinstance(dataset, MultiCountryDataSet):
            assert dataset_country is not None, "Must specify a country for multi country datasets"
            assert (
                dataset_country in dataset.countries
            ), f"Country {dataset_country} not found in dataset. Countries: {dataset.countries}"
            dataset = dataset[dataset_country]

    if "," in model_name:
        # model_name is not only one model, but contains a list of models
        model_list = model_name.split(",")
        model_configuration_yaml_list = [None for _ in model_list]
        if model_configuration_yaml is not None:
            model_configuration_yaml_list = model_configuration_yaml.split(",")
            assert len(model_list) == len(
                model_configuration_yaml_list
            ), "Number of model configurations does not match number of models"
    else:
        model_list = [model_name]
        model_configuration_yaml_list = [model_configuration_yaml]

    logger.info(f"Model configuration: {model_configuration_yaml_list}")

    results_dict = {}
    for name, configuration in zip(model_list, model_configuration_yaml_list):
        template = ModelTemplate.from_directory_or_github_url(
            name,
            base_working_dir=Path("./runs/"),
            ignore_env=ignore_environment,
            run_dir_type=run_directory_type,
        )
        logger.info(f"Model template loaded: {template}")
        if configuration is not None:
            logger.info(f"Loading model configuration from yaml file {configuration}")
            configuration = ModelConfiguration.model_validate(
                yaml.safe_load(open(configuration))
            )  # template.get_model_configuration_from_yaml(Path(configuration))
            logger.info(f"Loaded model configuration from yaml file: {configuration}")

        model = template.get_model(configuration)
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
        print("!!RESULTS:")
        print(results)
        results_dict[name] = results

    # need to iterate through the dict, like key and value or something and then extract the relevant metrics
    # to a pandas dataframe, and save it as a csv file.
    # it seems like results contain two dictionairies, one for aggregate metrics and one with seperate ones for each ts

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

    return results_dict


@app.command()
def sanity_check_model(
    model_url: str, use_local_environement: bool = False, dataset_path=None, model_config_path: str = None
):
    """
    Check that a model can be loaded, trained and used to make predictions
    """
    if dataset_path is None:
        dataset = datasets["hydromet_5_filtered"].load()
    else:
        dataset = DataSet.from_csv(dataset_path, FullData)
    train, tests = train_test_generator(dataset, 3, n_test_sets=2)
    context, future, truth = next(tests)
    logger.info("Dataset: ")
    logger.info(dataset.to_pandas())

    if model_config_path is not None:
        model_config = ModelConfiguration.model_validate(yaml.safe_load(open(model_config_path)))
    else:
        model_config = None
    try:
        model_template = get_model_template_from_directory_or_github_url(model_url, ignore_env=use_local_environement)
        model = model_template.get_model(
            model_config
        )  # get_model_from_directory_or_github_url(model_url, ignore_env=use_local_environement)
        estimator = model()
    except Exception as e:
        logger.error(f"Error while creating model: {e}")
        raise e
    try:
        predictor = estimator.train(train)
    except Exception as e:
        logger.error(f"Error while training model: {e}")
        raise e
    try:
        predictions = predictor.predict(context, future)
    except Exception as e:
        logger.error(f"Error while forecasting: {e}")
        raise e
    for location, prediction in predictions.items():
        assert not np.isnan(
            prediction.samples
        ).any(), f"NaNs in predictions for location {location}, {prediction.samples}"
    context, future, truth = next(tests)
    try:
        predictions = predictor.predict(context, future)
    except Exception as e:
        logger.error(f"Error while forecasting from a future time point: {e}")
        raise e
    for location, prediction in predictions.items():
        assert not np.isnan(
            prediction.samples
        ).any(), f"NaNs in futuresplit predictions for location {location}, {prediction.samples}"


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
    model = get_model_from_directory_or_github_url(model_name)
    model_name = model.name

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
def serve(seedfile: Optional[str] = None, debug: bool = False, auto_reload: bool = False):
    """
    Start CHAP as a backend server
    """
    from .rest_api_src.v1.rest_api import main_backend

    logger.info("Running chap serve")

    if seedfile is not None:
        data = json.load(open(seedfile))
    else:
        data = None

    main_backend(data, auto_reload=auto_reload)


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
    base_args = [("debug", bool, False), ("log_file", Optional[str], None)]

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
def plot_dataset(data_filename: Path, plot_name: str = "standardized_feature_plot"):
    dataset_plot_registry = {
        "standardized_feature_plot": StandardizedFeaturePlot,
        "season_plot": SeasonCorrelationBarPlot,
    }
    plot_cls = dataset_plot_registry.get(plot_name, StandardizedFeaturePlot)
    df = pd.read_csv(data_filename)
    plotter = plot_cls(df)
    fig = plotter.plot()
    fig.show()


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