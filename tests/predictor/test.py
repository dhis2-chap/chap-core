import pytest
import pandas as pd
from climate_health.predictor.poisson import Poisson


@pytest.fixture
def test_train():
    sample_data = pd.DataFrame({
        "Disease": [1, 2, 3, 4, 5],
        "Disease1": [1, 2, 3, 4, 5],
        "Disease2": [1, 2, 3, 4, 5],
        "Rain": [1, 2, 3, 4, 5],
        "Temperature": [1, 2, 3, 4, 5],
    })

    return Poisson().train(sample_data)


@pytest.fixture
def test_predict():
    sample_data = pd.DataFrame({
        "Disease1": [1, 2, 3, 4, 5],
        "Disease2": [1, 2, 3, 4, 5],
        "Rain": [1, 2, 3, 4, 5],
        "Temperature": [1, 2, 3, 4, 5],
    })

    return Poisson().predict(sample_data)
