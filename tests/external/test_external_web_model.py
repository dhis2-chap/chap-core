"""
Test the ExternalWebModel class with a real REST API server.
"""

import multiprocessing
import sys
import time
from pathlib import Path

import pandas as pd
import pytest
import requests
import uvicorn

from chap_core.datatypes import FullData
from chap_core.exceptions import ModelFailedException
from chap_core.models.external_web_model import ExternalWebModel
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

# Add the external_models directory to the path so we can import the API
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "external_models" / "web_based_model"))


def run_api_server():
    """Run the FastAPI server in a separate process."""
    from api import app

    uvicorn.run(app, host="127.0.0.1", port=8888, log_level="error")


@pytest.fixture(scope="module")
def api_server():
    """Start and stop the API server for testing."""
    # Start the API server in a separate process
    api_process = multiprocessing.Process(target=run_api_server)
    api_process.start()

    # Wait for the server to start
    api_url = "http://127.0.0.1:8888"
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{api_url}/health")
            if response.status_code == 200:
                break
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    else:
        api_process.terminate()
        api_process.join(timeout=5)
        if api_process.is_alive():
            api_process.kill()
            api_process.join()
        pytest.fail("API server failed to start")

    yield api_url

    # Teardown: stop the API server
    api_process.terminate()
    api_process.join(timeout=5)
    if api_process.is_alive():
        api_process.kill()
        api_process.join()


@pytest.fixture
def sample_data():
    """Create sample training and prediction data."""
    # Create combined training data with all required fields for FullData
    # FullData has: rainfall, mean_temperature, disease_cases, population
    train_data_df = pd.DataFrame(
        {
            "time_period": ["2023-01", "2023-02", "2023-03", "2023-04", "2023-05", "2023-06"],
            "location": ["loc1"] * 6,
            "disease_cases": [100, 120, 110, 130, 125, 140],
            "mean_temperature": [25.0, 26.0, 27.0, 28.0, 27.5, 26.5],
            "rainfall": [100, 150, 200, 180, 160, 120],
            "population": [10000] * 6,
        }
    )

    # Create future data for predictions (no disease_cases needed for future)
    future_data_df = pd.DataFrame(
        {
            "time_period": ["2023-07", "2023-08", "2023-09"],
            "location": ["loc1"] * 3,
            "mean_temperature": [26.0, 25.5, 25.0],
            "rainfall": [110, 130, 140],
            "population": [10000] * 3,
            "disease_cases": [0, 0, 0],  # Placeholder values for future
        }
    )

    # Convert to DataSet objects
    train_data = DataSet.from_pandas(train_data_df, FullData)
    future_data = DataSet.from_pandas(future_data_df, FullData)

    # Use train_data as historic_data for prediction
    historic_data = train_data

    return train_data, historic_data, future_data


@pytest.fixture
def model(api_server):
    """Create an ExternalWebModel instance."""
    return ExternalWebModel(
        api_url=api_server,
        name="test_model",
        timeout=30,
        poll_interval=0.5,
    )


def test_model_initialization(api_server):
    """Test that we can initialize the model."""
    model = ExternalWebModel(
        api_url=api_server,
        name="test_model",
        timeout=60,
        poll_interval=1,
    )

    assert model.name == "test_model"
    assert model._api_url == api_server
    assert model._timeout == 60
    assert model._poll_interval == 1


def test_health_check(api_server):
    """Test that the API health check works."""
    response = requests.get(f"{api_server}/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_train_and_predict(model, sample_data):
    """Test the full train and predict workflow."""
    train_data, historic_data, future_data = sample_data

    # Train the model
    trained_model = model.train(train_data)
    assert trained_model is model
    assert model._trained_model_name is not None

    # Make predictions
    predictions = model.predict(historic_data, future_data)

    # Verify predictions
    assert predictions is not None
    predictions_df = predictions.to_pandas()
    assert len(predictions_df) > 0
    assert "time_period" in predictions_df.columns
    assert "location" in predictions_df.columns


def test_predict_without_training_fails(api_server, sample_data):
    """Test that prediction without training fails."""
    model = ExternalWebModel(
        api_url=api_server,
        name="test_model",
        timeout=10,
        poll_interval=0.5,
    )

    _, historic_data, future_data = sample_data

    with pytest.raises(ModelFailedException) as exc_info:
        model.predict(historic_data, future_data)

    assert "must be trained" in str(exc_info.value)


def test_job_status_tracking(model, sample_data):
    """Test that job status is properly tracked."""
    train_data, _, _ = sample_data

    # Train and verify the model name was set
    model.train(train_data)
    assert model._trained_model_name is not None
    assert "test_model" in model._trained_model_name


def test_with_configuration(api_server, sample_data):
    """Test that configuration is properly passed to the API."""
    config = {
        "feature_columns": ["temperature", "rainfall"],
        "model_type": "linear",
        "hyperparameters": {
            "learning_rate": 0.01,
            "epochs": 100,
        },
    }

    model = ExternalWebModel(
        api_url=api_server,
        name="configured_model",
        timeout=30,
        poll_interval=0.5,
        configuration=config,
    )

    assert model.configuration == config

    train_data, _, _ = sample_data
    model.train(train_data)
    assert model._trained_model_name is not None


def test_multiple_models_can_be_trained(api_server, sample_data):
    """Test that multiple models can be trained independently."""
    train_data, historic_data, future_data = sample_data

    # Create and train first model
    model1 = ExternalWebModel(
        api_url=api_server,
        name="model1",
        timeout=30,
        poll_interval=0.5,
    )
    model1.train(train_data)

    # Create and train second model
    model2 = ExternalWebModel(
        api_url=api_server,
        name="model2",
        timeout=30,
        poll_interval=0.5,
    )
    model2.train(train_data)

    # Both models should have different trained names
    assert model1._trained_model_name != model2._trained_model_name
    assert "model1" in model1._trained_model_name
    assert "model2" in model2._trained_model_name

    # Both should be able to predict
    predictions1 = model1.predict(historic_data, future_data)
    predictions2 = model2.predict(historic_data, future_data)

    assert predictions1 is not None
    assert predictions2 is not None


if __name__ == "__main__":
    # Allow running the test directly
    pytest.main([__file__, "-v"])
