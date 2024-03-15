from .mock_predictor_script import predict_values
import tempfile


def test():
    # testing that nothing crashes
    train_file_name = "tests/mock_predictor_script_train.csv"
    future_climate_file_name = "tests/mock_predictor_script_future_climate.csv"

    with tempfile.NamedTemporaryFile() as out:
        predict_values(train_file_name, future_climate_file_name, out.name)
