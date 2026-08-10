import logging
import os

import mlflow.exceptions
import mlflow.projects
from mlflow.utils.process import ShellCommandException

from chap_core.exceptions import ModelFailedException
from chap_core.runners.runner import TrainPredictRunner

logger = logging.getLogger(__name__)


def _set_default_tracking_uri():
    """Point MLflow at a writable tracking store when the deployment has not configured one.

    MLflow defaults its tracking store to the relative URI ``sqlite:///mlflow.db``, which it
    resolves against the process working directory. In the worker container that directory is
    ``/app`` on a read-only filesystem, so creating a run fails with "unable to open database
    file". CHAP_RUNS_DIR is writable in every deployment, so use it as the fallback location.
    """
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return

    from chap_core.models.utils import CHAP_RUNS_DIR

    db_path = (CHAP_RUNS_DIR / "mlflow.db").absolute()
    logger.debug(f"MLFLOW_TRACKING_URI not set, defaulting MLflow tracking store to {db_path}")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")


class MlFlowTrainPredictRunner(TrainPredictRunner):
    def __init__(self, model_path, model_configuration_filename=None, train_params=None):
        _set_default_tracking_uri()
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
        return mlflow.projects.run(
            str(self.model_path),
            entry_point="predict",
            parameters=params,
        )

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
