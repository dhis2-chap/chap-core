from chap_core.exceptions import ModelFailedException
from chap_core.runners.runner import TrainPredictRunner
import mlflow.exceptions
import mlflow.projects
from mlflow.utils.process import ShellCommandException
import logging

logger = logging.getLogger(__name__)


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
        logger.info("Training model using MLflow")
        try:
            keys = {"train_data": str(train_file_name), "model": str(model_file_name)}
            possible_extra = {
                "model_config": str(self.model_configuration_filename) if self.model_configuration_filename else None,
            }
            keys.update({key: val for key, val in possible_extra.items() if key in self.extra_params})
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
        if self.model_configuration_filename is not None:
            ("Model configuration not supported for MLflow runner")
        params = {
            "historic_data": str(historic_data),
            "future_data": str(future_data),
            "model": str(model_file_name),
            "out_file": str(output_file),
        }
        extra_params = {
            "model_config": str(self.model_configuration_filename) if self.model_configuration_filename else None,
        }
        params.update({key: val for key, val in extra_params.items() if key in self.extra_params})
        return mlflow.projects.run(
            str(self.model_path),
            entry_point="predict",
            parameters=params,
        )
