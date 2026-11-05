import logging
import os
from contextlib import contextmanager
from pathlib import Path

import mlflow.exceptions
import mlflow.projects
import mlflow.tracking
from mlflow.utils.process import ShellCommandException

from chap_core.exceptions import ModelFailedException
from chap_core.runners.runner import TrainPredictRunner

logger = logging.getLogger(__name__)


@contextmanager
def tracking_in_working_directory(working_dir):
    """Keep MLflow's run records in the model's working directory, unless a tracking URI is set.

    ``mlflow.projects.run`` records every run in the tracking store, which defaults to
    ``mlflow.db`` in the current directory, so running a model would leave that file wherever
    chap was started, such as in a Chap project. An ``MLFLOW_TRACKING_URI`` you set is used as is.
    """
    if mlflow.tracking.is_tracking_uri_set():
        yield
        return
    database = Path(working_dir).resolve() / "mlflow.db"
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{database.as_posix()}"
    try:
        yield
    finally:
        os.environ.pop("MLFLOW_TRACKING_URI", None)


class MlFlowTrainPredictRunner(TrainPredictRunner):
    def __init__(self, model_path, model_configuration_filename=None, train_params=None):
        self.model_path = model_path
        self.model_configuration_filename = model_configuration_filename

        # This logic should probably be a better
        # Find out which parameters are used in the MLproject file
        # Assumes now that the extra parameters are the same in train and predict
        if train_params is None:
            self.extra_params = []
        else:
            self.extra_params = [key for key in train_params if key not in ["train_data", "model"]]

    def train(self, train_file_name, model_file_name, polygons_file_name=None):
        try:
            # train_file_name = Path(self.model_path) /  Path(train_file_name)
            keys = {"train_data": str(train_file_name), "model": str(model_file_name)}
            logger.info(f"Training model using MLflow, working dir is {self.model_path}. Train data: {keys}")
            possible_extra = {
                "model_config": str(self.model_configuration_filename) if self.model_configuration_filename else None,
            }
            keys.update(
                {key: val for key, val in possible_extra.items() if key in self.extra_params and val is not None}
            )
            with tracking_in_working_directory(self.model_path):
                return mlflow.projects.run(
                    str(self.model_path),
                    entry_point="train",
                    parameters=keys,
                    build_image=True,
                )
        except ShellCommandException as e:
            logger.error(
                "Error running mlflow project, might be due to missing pyenv (See: https://github.com/pyenv/pyenv#installation)"
            )
            raise ModelFailedException(str(e)) from e
        except mlflow.exceptions.ExecutionException as e:
            logger.error("Executation of model failed for some reason. Check the logs for more information")
            raise ModelFailedException(str(e)) from e

    def predict(self, model_file_name, historic_data, future_data, output_file, polygons_file_name=None):
        logging.debug(f"Running predict with output to {output_file}")
        if self.model_configuration_filename is not None:
            logger.warning("Model configuration not supported for MLflow runner")
        params = {
            "historic_data": str(historic_data),
            "future_data": str(future_data),
            "model": str(model_file_name),
            "out_file": str(output_file),
        }
        logging.debug(f"Params for predict: {params}")
        extra_params = {
            "model_config": str(self.model_configuration_filename) if self.model_configuration_filename else None,
        }
        params.update({key: val for key, val in extra_params.items() if key in self.extra_params and val is not None})
        try:
            with tracking_in_working_directory(self.model_path):
                return mlflow.projects.run(
                    str(self.model_path),
                    entry_point="predict",
                    parameters=params,
                )
        except ShellCommandException as e:
            logger.error(
                "Error running mlflow project, might be due to missing pyenv (See: https://github.com/pyenv/pyenv#installation)"
            )
            raise ModelFailedException(str(e)) from e
        except mlflow.exceptions.ExecutionException as e:
            logger.error("Execution of model failed for some reason. Check the logs for more information")
            raise ModelFailedException(str(e)) from e

    def report(self, model_file_name, historic_data, output_file, polygons_file_name=None):
        logging.debug(f"Running report with output to {output_file}")
        params = {
            "historic_data": str(historic_data),
            "model": str(model_file_name),
            "out_file": str(output_file),
        }
        extra_params = {
            "model_config": str(self.model_configuration_filename) if self.model_configuration_filename else None,
        }
        params.update({key: val for key, val in extra_params.items() if key in self.extra_params and val is not None})
        try:
            with tracking_in_working_directory(self.model_path):
                return mlflow.projects.run(
                    str(self.model_path),
                    entry_point="report",
                    parameters=params,
                )
        except ShellCommandException as e:
            logger.error(
                "Error running mlflow project, might be due to missing pyenv (See: https://github.com/pyenv/pyenv#installation)"
            )
            raise ModelFailedException(str(e)) from e
        except mlflow.exceptions.ExecutionException as e:
            logger.error("Execution of model failed for some reason. Check the logs for more information")
            raise ModelFailedException(str(e)) from e
