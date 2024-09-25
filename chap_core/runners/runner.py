class Runner:
    def run_command(self, command): ...

    def store_file(self, file_path): ...


class TrainPredictRunner:
    """
    Specific wrapper for runners that only run train/predict commands
    """

    def train(self, train_data: str, model_file_name: str): ...

    def predict(
        self,
        model_file_name: str,
        historic_data: str,
        future_data: str,
        output_file: str,
    ): ...
