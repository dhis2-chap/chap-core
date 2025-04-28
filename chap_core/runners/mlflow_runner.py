from chap_core.exceptions import ModelFailedException
from chap_core.runners.runner import TrainPredictRunner
import mlflow.exceptions
import mlflow.projects
from mlflow.utils.process import ShellCommandException
import logging

logger = logging.getLogger(__name__)


class MlFlowTrainPredictRunner(TrainPredictRunner):
    def __init__(self, model_path):
        self.model_path = model_path

    def train(self, train_file_name, model_file_name, polygons_file_name=None):
        logger.info("Training model using MLflow")
        try:
            return mlflow.projects.run(
                str(self.model_path),
                entry_point="train",
                parameters={
                    "train_data": str(train_file_name),
                    "model": str(model_file_name),
                },
                build_image=True,
            )
        except ShellCommandException as e:
            logger.error(
                "Error running mlflow project, might be due to missing pyenv (See: https://github.com/pyenv/pyenv#installation)"
            )
            raise ModelFailedException(str(e))
        except mlflow.exceptions.ExecutionException as e:
            logger.error(
                "Executation of model failed for some reason. Check the logs for more information"
            )
            raise ModelFailedException(str(e))

    def predict(self, model_file_name, historic_data, future_data, output_file, polygons_file_name=None):
        return mlflow.projects.run(
            str(self.model_path),
            entry_point="predict",
            parameters={
                "historic_data": str(historic_data),
                "future_data": str(future_data),
                "model": str(model_file_name),
                "out_file": str(output_file),
            },
        )