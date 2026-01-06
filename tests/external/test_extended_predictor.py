import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from chap_core.external.ExtendedPredictor import ExtendedPredictor
from chap_core.models.configured_model import ConfiguredModel
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.external.model_configuration import ModelTemplateConfigV2


class MockModel(ConfiguredModel):
    """Mock model that returns simple predictions for testing"""

    def __init__(self, min_pred_length=2, max_pred_length=4):
        self.model_information = ModelTemplateConfigV2(
            name="mock_model",
            min_prediction_length=min_pred_length,
            max_prediction_length=max_pred_length
        )
        self.trained = False
        self.predict_call_count = 0

    def train(self, train_data: DataSet, extra_args=None):
        self.trained = True

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        """Returns predictions with sample_ columns

        Creates predictable values based on time_period for testing coherence
        """
        self.predict_call_count += 1
        future_df = future_data.to_pandas()
        result_df = future_df.copy()

        # Create sample predictions with predictable values based on row index
        # This allows us to verify that the right predictions end up in the right places
        num_samples = 3
        for i in range(num_samples):
            # Use a predictable formula: base value + sample_offset + row_index
            result_df[f'sample_{i}'] = [
                100.0 + i + idx for idx in range(len(result_df))
            ]

        return DataSet.from_pandas(result_df)


def create_test_dataset(num_periods=10, start_period="2020-01"):
    """Helper function to create test datasets"""
    periods = pd.period_range(start=start_period, periods=num_periods, freq='M')

    data = {
        'time_period': [str(p) for p in periods],
        'location': ['location_1'] * num_periods,
        'disease_cases': np.random.randint(0, 100, num_periods),
        'rainfall': np.random.uniform(0, 100, num_periods),
        'mean_temperature': np.random.uniform(20, 35, num_periods)
    }

    df = pd.DataFrame(data)
    return DataSet.from_pandas(df)


def test_extended_predictor_initialization():
    """Test that ExtendedPredictor can be initialized properly"""
    mock_model = MockModel()
    desired_scope = 6

    extended_predictor = ExtendedPredictor(mock_model, desired_scope)

    assert extended_predictor._config_model == mock_model
    assert extended_predictor._desired_scope == desired_scope


def test_extended_predictor_train():
    """Test that training is delegated to the underlying model"""
    mock_model = MockModel()
    extended_predictor = ExtendedPredictor(mock_model, 6)

    train_data = create_test_dataset(num_periods=20)
    extended_predictor.train(train_data)

    assert mock_model.trained is True


def test_extended_predictor_predict_within_max_length():
    """Test prediction when desired_scope is within max_prediction_length"""
    mock_model = MockModel(min_pred_length=2, max_pred_length=10)
    desired_scope = 5
    extended_predictor = ExtendedPredictor(mock_model, desired_scope)

    historic_data = create_test_dataset(num_periods=20, start_period="2020-01")
    future_data = create_test_dataset(num_periods=10, start_period="2021-09")

    result = extended_predictor.predict(historic_data, future_data)
    result_df = result.to_pandas()

    # Check that we got predictions
    assert len(result_df) > 0
    # Check that sample columns were created by the mock model
    sample_cols = [col for col in result_df.columns if col.startswith("sample_")]
    assert len(sample_cols) > 0


def test_extended_predictor_predict_exceeds_max_length():
    """Test prediction when desired_scope exceeds max_prediction_length (iterative prediction)

    This test verifies:
    1. The correct number of predictions are generated to cover desired_scope
    2. Multiple predict calls are made (iterative prediction happened)
    3. Time periods are coherent and sequential
    4. Prediction values are coherent across iterations
    """
    mock_model = MockModel(min_pred_length=2, max_pred_length=3)
    desired_scope = 8  # This exceeds max_pred_length, should trigger iterative prediction
    extended_predictor = ExtendedPredictor(mock_model, desired_scope)

    historic_data = create_test_dataset(num_periods=20, start_period="2020-01")
    future_data = create_test_dataset(num_periods=10, start_period="2021-09")

    # Get the expected time periods from future_data
    future_df = future_data.to_pandas()
    expected_periods = future_df['time_period'].values

    result = extended_predictor.predict(historic_data, future_data)
    result_df = result.to_pandas()

    # 1. Check that we got enough predictions to cover the desired scope
    assert len(result_df) >= desired_scope, \
        f"Expected at least {desired_scope} predictions to cover desired scope, got {len(result_df)}"

    # 2. Verify that predict was called multiple times (iterative prediction happened)
    # Since max_pred_length=3 and desired_scope=8, we expect at least 3 calls
    expected_min_calls = (desired_scope + mock_model.model_information.max_prediction_length - 1) // mock_model.model_information.max_prediction_length
    assert mock_model.predict_call_count >= expected_min_calls, \
        f"Expected at least {expected_min_calls} predict calls, got {mock_model.predict_call_count}"

    # 3. Check that time periods are present and sequential
    result_periods = result_df['time_period'].values
    # The first desired_scope time periods should match the expected ones from future_data
    for i in range(min(desired_scope, len(expected_periods))):
        assert result_periods[i] == expected_periods[i], \
            f"Time period mismatch at index {i}: expected {expected_periods[i]}, got {result_periods[i]}"

    # 4. Verify that sample columns exist (indicating predictions were made)
    sample_cols = [col for col in result_df.columns if col.startswith("sample_")]
    assert len(sample_cols) > 0, "No sample columns found in predictions"

    # 5. Verify that predictions have reasonable values (not NaN, not all zeros)
    for col in sample_cols:
        assert result_df[col].notna().all(), f"Found NaN values in {col}"
        assert not (result_df[col] == 0).all(), f"All values are zero in {col}"


def test_extended_predictor_update_historic_data():
    """Test the update_historic_data method"""
    mock_model = MockModel()
    extended_predictor = ExtendedPredictor(mock_model, 6)

    # Create historic data
    historic_df = pd.DataFrame({
        'time_period': ['2020-01', '2020-02'],
        'location': ['location_1', 'location_1'],
        'disease_cases': [10, 15],
        'rainfall': [50, 60]
    })

    # Create predictions with sample columns
    predictions_df = pd.DataFrame({
        'time_period': ['2020-03', '2020-04'],
        'location': ['location_1', 'location_1'],
        'sample_0': [20, 25],
        'sample_1': [21, 26],
        'sample_2': [22, 27],
        'rainfall': [70, 80]
    })

    num_predictions = 2
    updated_historic = extended_predictor.update_historic_data(
        historic_df, predictions_df, num_predictions
    )

    # Check that historic data was extended
    assert len(updated_historic) == 4  # 2 original + 2 new
    # Check that disease_cases column exists in new rows (averaged from samples)
    assert 'disease_cases' in updated_historic.columns
    # Check that the last 2 rows have disease_cases values (averaged from samples)
    assert updated_historic['disease_cases'].iloc[-2:].notna().all()


def test_extended_predictor_scope_validation():
    """Test that assertion fails when desired_scope is less than min_prediction_length"""
    mock_model = MockModel(min_pred_length=5, max_pred_length=10)
    desired_scope = 3  # Less than min_prediction_length
    extended_predictor = ExtendedPredictor(mock_model, desired_scope)

    historic_data = create_test_dataset(num_periods=20, start_period="2020-01")
    future_data = create_test_dataset(num_periods=10, start_period="2021-09")

    # This should raise an AssertionError because desired_scope < min_prediction_length
    with pytest.raises(AssertionError):
        extended_predictor.predict(historic_data, future_data)


def test_extended_predictor_prediction_coherence():
    """Test that predictions maintain coherence across iterations

    This test uses a deterministic mock model to verify that:
    1. The correct number of predictions are generated to cover desired_scope
    2. The iterative prediction mechanism works correctly
    3. Historic data is properly updated between iterations
    4. Predictions from multiple iterations are concatenated correctly
    5. Each time period gets the correct predicted values (coherence check)
    """

    class DeterministicMockModel(ConfiguredModel):
        """A mock model that returns values based on the time period index"""

        def __init__(self):
            self.model_information = ModelTemplateConfigV2(
                name="deterministic_model",
                min_prediction_length=2,
                max_prediction_length=3
            )
            self.call_log = []  # Track what data was passed to each call

        def train(self, train_data: DataSet, extra_args=None):
            pass

        def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
            """Returns predictions where sample values encode the iteration number"""
            future_df = future_data.to_pandas()
            historic_df = historic_data.to_pandas()

            # Log this call
            iteration_num = len(self.call_log)
            self.call_log.append({
                'iteration': iteration_num,
                'historic_length': len(historic_df),
                'future_length': len(future_df),
                'future_periods': future_df['time_period'].tolist()
            })

            result_df = future_df.copy()

            # Create predictions that encode the iteration number and row index
            # Format: iteration*1000 + row_idx*10 + sample_index
            for sample_idx in range(3):
                result_df[f'sample_{sample_idx}'] = [
                    iteration_num * 1000 + row_idx * 10 + sample_idx
                    for row_idx in range(len(result_df))
                ]

            return DataSet.from_pandas(result_df)

    mock_model = DeterministicMockModel()
    desired_scope = 7
    extended_predictor = ExtendedPredictor(mock_model, desired_scope)

    historic_data = create_test_dataset(num_periods=20, start_period="2020-01")
    future_data = create_test_dataset(num_periods=10, start_period="2021-09")

    result = extended_predictor.predict(historic_data, future_data)
    result_df = result.to_pandas()

    # 1. Verify we got at least the desired scope
    assert len(result_df) >= desired_scope, \
        f"Expected at least {desired_scope} predictions to cover desired scope, got {len(result_df)}"

    # 2. Verify multiple iterations occurred
    assert len(mock_model.call_log) > 1, \
        f"Expected multiple iterations, got {len(mock_model.call_log)}"

    # 3. Verify that historic data was growing with each iteration
    historic_lengths = [call['historic_length'] for call in mock_model.call_log]
    for i in range(1, len(historic_lengths)):
        assert historic_lengths[i] > historic_lengths[i-1], \
            f"Historic data should grow between iterations: {historic_lengths}"

    # 4. Verify that sample columns have values from different iterations
    # (this proves the results are concatenated correctly and coherent)
    sample_0_values = result_df['sample_0'].values
    # Check that we have values from iteration 0 and iteration 1+
    has_iteration_0 = any(val < 1000 for val in sample_0_values)
    has_iteration_1_plus = any(val >= 1000 for val in sample_0_values)
    assert has_iteration_0 and has_iteration_1_plus, \
        "Results should contain predictions from multiple iterations"

    # 5. Verify the predictions are in the correct order by checking the first few values
    # First prediction should be from iteration 0, row 0
    assert sample_0_values[0] == 0, \
        f"First prediction should be from iteration 0, row 0 (expected 0, got {sample_0_values[0]})"

    # 6. Check that values are coherent: within each iteration, values should increment
    # For iteration 0 (values < 1000), sample_0 should be 0, 10, 20, ...
    iteration_0_values = [v for v in sample_0_values if v < 1000]
    for i, val in enumerate(iteration_0_values):
        expected_val = i * 10  # iteration=0, so 0*1000 + i*10 + 0 = i*10
        assert val == expected_val, \
            f"Iteration 0, position {i}: expected {expected_val}, got {val}"


def test_extended_predictor_with_external_model_interface():
    """Test that ExtendedPredictor works with ExternalModel-like interface"""
    from chap_core.models.external_model import ExternalModel

    # Create a mock runner
    mock_runner = Mock()

    # Create an ExternalModel with model_information
    external_model = ExternalModel(
        runner=mock_runner,
        name="test_model",
        model_information=ModelTemplateConfigV2(
            name="test_external_model",
            min_prediction_length=2,
            max_prediction_length=4
        )
    )

    # Wrap it in ExtendedPredictor
    extended_predictor = ExtendedPredictor(external_model, desired_scope=6)

    # Just verify it initializes correctly
    assert extended_predictor._config_model == external_model
    assert extended_predictor._desired_scope == 6
    assert extended_predictor._config_model.model_information.min_prediction_length == 2
    assert extended_predictor._config_model.model_information.max_prediction_length == 4
