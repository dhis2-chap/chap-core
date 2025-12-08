"""Utility commands for CHAP CLI."""

import dataclasses
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.datatypes import FullData
from chap_core.log_config import initialize_logging
from chap_core.models.utils import get_model_template_from_directory_or_github_url
from chap_core.plotting.dataset_plot import StandardizedFeaturePlot
from chap_core.plotting.season_plot import SeasonCorrelationBarPlot
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.file_io.example_data_set import datasets

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class AreaPolygons: ...


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
        model = model_template.get_model(model_config)
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
        assert not np.isnan(prediction.samples).any(), (
            f"NaNs in predictions for location {location}, {prediction.samples}"
        )
    context, future, truth = next(tests)
    try:
        predictions = predictor.predict(context, future)
    except Exception as e:
        logger.error(f"Error while forecasting from a future time point: {e}")
        raise e
    for location, prediction in predictions.items():
        assert not np.isnan(prediction.samples).any(), (
            f"NaNs in futuresplit predictions for location {location}, {prediction.samples}"
        )


def serve(seedfile: Optional[str] = None, debug: bool = False, auto_reload: bool = False):
    """
    Start CHAP as a backend server
    """
    from chap_core.rest_api_src.v1.rest_api import main_backend

    logger.info("Running chap serve")

    if seedfile is not None:
        data = json.load(open(seedfile))
    else:
        data = None

    main_backend(data, auto_reload=auto_reload)


def write_open_api_spec(out_path: str):
    """
    Write the OpenAPI spec to a file
    """
    from chap_core.rest_api_src.v1.rest_api import get_openapi_schema

    schema = get_openapi_schema()
    with open(out_path, "w") as f:
        json.dump(schema, f, indent=4)


def test(**base_kwargs):
    """
    Simple test-command to check that the chap command works
    """
    initialize_logging()

    logger.debug("Debug message")
    logger.info("Info message")


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


def register_commands(app):
    """Register utility commands with the CLI app."""
    app.command()(sanity_check_model)
    app.command()(serve)
    app.command()(write_open_api_spec)
    app.command()(test)
    app.command()(plot_dataset)
