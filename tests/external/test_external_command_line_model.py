import logging

import pytest

logging.basicConfig(level=logging.INFO)
from climate_health.datatypes import ClimateHealthData
from climate_health.external.external_model import ExternalCommandLineModel
from ..data_fixtures import train_data, full_data, future_climate_data

@pytest.mark.xfail(reason='fails')
def test(train_data, future_climate_data):
    train_command = "bash ./tests/external/command_line_model_train.sh {train_data} {model}"
    predict_command = "bash ./tests/external/command_line_model_predict.sh {future_data} {model} {out_file}"
    conda_env = "tests/external/external_model_env.yml"
    conda_env = ""  # requires conda if using conda env

    model = ExternalCommandLineModel("test_model", train_command, predict_command, data_type=ClimateHealthData,
                                     conda_env_file=conda_env)
    model.setup()
    model.train(train_data)
    predictions = model.predict(future_climate_data)

    print(predictions)


if __name__ == "__main__":
    test(train_data(), future_climate_data())
