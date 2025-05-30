import abc
from typing import Optional


class Runner:
    """
    An interface for Runners. A runner is able to run "something", e.g. a command on the command line
    through Docker."""

    def run_command(self, command): ...

    def store_file(self, file_path):
        ...
        # not used for anything now

    def teardown(self):
        """To be called after the runner is done with train and predict. This is to clean up the runner, e.g.
        to remove docker images, etc"""
        ...


class TrainPredictRunner(abc.ABC):
    """
    Specific wrapper for runners that only run train/predict commands
    """

    @abc.abstractmethod
    def train(self, train_data: str, model_file_name: str, polygons_file_name: Optional[str]): ...

    @abc.abstractmethod
    def predict(
        self,
        model_file_name: str,
        historic_data: str,
        future_data: str,
        output_file: str,
        polygons_file_name: Optional[str],
    ): ...

    def teardown(self): ...
