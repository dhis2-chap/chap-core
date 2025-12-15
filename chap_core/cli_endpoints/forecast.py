"""Forecast commands for CHAP CLI."""

import logging
from pathlib import Path
from typing import Optional

from chap_core.assessment.forecast import multi_forecast as do_multi_forecast
from chap_core.models.utils import get_model_from_directory_or_github_url
from chap_core.plotting.prediction_plot import plot_forecast_from_summaries
from chap_core import api
from chap_core.file_io.example_data_set import datasets, DataSetType
from chap_core.time_period.date_util_wrapper import delta_month

logger = logging.getLogger(__name__)


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


def register_commands(app):
    """Register forecast commands with the CLI app."""
    app.command()(forecast)
    app.command()(multi_forecast)
