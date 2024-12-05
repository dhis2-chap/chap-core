from typing import Optional


class Runner:
    def run_command(self, command): ...

    def store_file(self, file_path): ...

    def teardown(self):
        """To be called after the runner is done with train and predict. This is to clean up the runner, e.g.
        to remove docker images, etc"""
        ...


class TrainPredictRunner:
    """
    Specific wrapper for runners that only run train/predict commands
    """

    def train(self, train_data: str, model_file_name: str, polygons_file_name: Optional[str]): ...

    def predict(
        self,
        model_file_name: str,
        historic_data: str,
        future_data: str,
        output_file: str,
        polygons_file_name: Optional[str],
    ): ...

    def teardown(self): ...

