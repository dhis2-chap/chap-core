import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from chap_core.external.ExtendedPredictor import ExtendedPredictor
from chap_core.models.configured_model import ConfiguredModel
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.external.model_configuration import ModelTemplateConfigV2


class MockModel(ConfiguredModel):
    """Mock model that returns simple predictions for testing."""

    def __init__(self, min_pred_length=2, max_pred_length=4):
        self.model_information = ModelTemplateConfigV2(
            name="mock_model", min_prediction_length=min_pred_length, max_prediction_length=max_pred_length
        )
        self.trained = False
        self.predict_call_count = 0

    def train(self, train_data: DataSet, extra_args=None):
        self.trained = True
        return self

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        """Returns predictions with sample_ columns."""
        self.predict_call_count += 1
        future_df = future_data.to_pandas()
        result_df = future_df.copy()

        num_samples = 3
        for i in range(num_samples):
            result_df[f"sample_{i}"] = [100.0 + i + idx for idx in range(len(result_df))]

        return DataSet.from_pandas(result_df)  # type: ignore[reportArgumentType]


class TrackingMockModel(ConfiguredModel):
    """Mock model that tracks inputs and produces verifiable outputs.

    This model records every predict() call so tests can verify:
    - What historic/future data was passed to each iteration
    - That all locations receive correct predictions

    Prediction values encode location and period indices for verification.
    """

    def __init__(self, locations, future_periods, min_pred_length=2, max_pred_length=3):
        self.model_information = ModelTemplateConfigV2(
            name="tracking_mock",
            min_prediction_length=min_pred_length,
            max_prediction_length=max_pred_length,
        )
        self.locations = locations
        self.future_periods = future_periods
        self.call_history = []

    def train(self, train_data: DataSet, extra_args=None):
        return self

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        historic_df = historic_data.to_pandas()
        future_df = future_data.to_pandas()

        call_num = len(self.call_history)

        self.call_history.append(
            {
                "call_num": call_num,
                "historic_locations": set(historic_df["location"].unique()),
                "historic_periods_per_loc": {
                    loc: sorted(historic_df[historic_df["location"] == loc]["time_period"].tolist())
                    for loc in historic_df["location"].unique()
                },
                "future_locations": set(future_df["location"].unique()),
                "future_periods_per_loc": {
                    loc: sorted(future_df[future_df["location"] == loc]["time_period"].tolist())
                    for loc in future_df["location"].unique()
                },
            }
        )

        result_rows = []
        for _, row in future_df.iterrows():
            loc_idx = self.locations.index(row["location"])
            period_str = str(row["time_period"])
            period_idx = self.future_periods.index(period_str)
            result_row = {
                "time_period": row["time_period"],
                "location": row["location"],
                # Encode: call_num * 10000 + loc_idx * 100 + period_idx
                "sample_0": call_num * 10000 + loc_idx * 100 + period_idx,
                "sample_1": call_num * 10000 + loc_idx * 100 + period_idx + 0.1,
                "sample_2": call_num * 10000 + loc_idx * 100 + period_idx + 0.2,
            }
            result_rows.append(result_row)

        return DataSet.from_pandas(pd.DataFrame(result_rows))  # type: ignore[reportArgumentType]


def create_test_dataset(num_periods=10, start_period="2020-01"):
    """Helper function to create test datasets with a single location."""
    periods = pd.period_range(start=start_period, periods=num_periods, freq="M")

    data = {
        "time_period": [str(p) for p in periods],
        "location": ["location_1"] * num_periods,
        "disease_cases": np.random.randint(0, 100, num_periods),
        "rainfall": np.random.uniform(0, 100, num_periods),
        "mean_temperature": np.random.uniform(20, 35, num_periods),
    }

    df = pd.DataFrame(data)
    return DataSet.from_pandas(df)  # type: ignore[reportArgumentType]


def create_multi_location_data(num_locations=4, num_historic_periods=10, num_future_periods=8):
    """Create deterministic test data with multiple locations."""
    locations = [f"loc_{i}" for i in range(num_locations)]
    all_periods = pd.period_range(start="2020-01", periods=num_historic_periods + num_future_periods, freq="M")
    historic_periods = [str(p) for p in all_periods[:num_historic_periods]]
    future_periods = [str(p) for p in all_periods[num_historic_periods:]]

    historic_rows = []
    for loc in locations:
        for period in historic_periods:
            historic_rows.append({"time_period": period, "location": loc, "disease_cases": 100, "rainfall": 50.0})
    historic_data = DataSet.from_pandas(pd.DataFrame(historic_rows))  # type: ignore[reportArgumentType]

    future_rows = []
    for loc in locations:
        for period in future_periods:
            future_rows.append({"time_period": period, "location": loc, "rainfall": 60.0})
    future_data = DataSet.from_pandas(pd.DataFrame(future_rows))  # type: ignore[reportArgumentType]

    return {
        "locations": locations,
        "historic_periods": historic_periods,
        "future_periods": future_periods,
        "historic_data": historic_data,
        "future_data": future_data,
    }


def test_extended_predictor_initialization():
    """Test that ExtendedPredictor can be initialized properly."""
    mock_model = MockModel()
    desired_scope = 6

    extended_predictor = ExtendedPredictor(mock_model, desired_scope)

    assert extended_predictor._config_model == mock_model
    assert extended_predictor._desired_scope == desired_scope


def test_extended_predictor_train():
    """Test that training is delegated to the underlying model and returns self."""
    mock_model = MockModel()
    extended_predictor = ExtendedPredictor(mock_model, 6)

    train_data = create_test_dataset(num_periods=20)
    result = extended_predictor.train(train_data)

    assert mock_model.trained is True
    assert result is extended_predictor, "train() must return self for evaluate_model compatibility"


def test_extended_predictor_scope_validation():
    """Test that assertion fails when desired_scope is less than min_prediction_length."""
    mock_model = MockModel(min_pred_length=5, max_pred_length=10)
    desired_scope = 3
    extended_predictor = ExtendedPredictor(mock_model, desired_scope)

    historic_data = create_test_dataset(num_periods=20, start_period="2020-01")
    future_data = create_test_dataset(num_periods=10, start_period="2021-09")

    with pytest.raises(AssertionError):
        extended_predictor.predict(historic_data, future_data)


def test_extended_predictor_with_external_model_interface():
    """Test that ExtendedPredictor works with ExternalModel-like interface."""
    from chap_core.models.external_model import ExternalModel

    mock_runner = Mock()

    external_model = ExternalModel(
        runner=mock_runner,
        name="test_model",
        model_information=ModelTemplateConfigV2(
            name="test_external_model", min_prediction_length=2, max_prediction_length=4
        ),
    )

    extended_predictor = ExtendedPredictor(external_model, desired_scope=6)

    assert extended_predictor._config_model == external_model
    assert extended_predictor._desired_scope == 6
    assert extended_predictor._config_model.model_information.min_prediction_length == 2  # type: ignore[reportAttributeAccessIssue]
    assert extended_predictor._config_model.model_information.max_prediction_length == 4  # type: ignore[reportAttributeAccessIssue]


def test_update_historic_data_includes_all_locations():
    """Test that update_historic_data adds predictions for all locations.

    Regression test for bug where .head(num_predictions) only took rows
    instead of time periods, missing data for other locations.
    """
    mock_model = MockModel()
    extended_predictor = ExtendedPredictor(mock_model, 6)

    historic_df = pd.DataFrame(
        {
            "time_period": ["2020-01", "2020-02", "2020-01", "2020-02"],
            "location": ["loc_A", "loc_A", "loc_B", "loc_B"],
            "disease_cases": [10, 15, 20, 25],
            "rainfall": [50, 60, 55, 65],
        }
    )

    predictions_df = pd.DataFrame(
        {
            "time_period": ["2020-03", "2020-04", "2020-03", "2020-04"],
            "location": ["loc_A", "loc_A", "loc_B", "loc_B"],
            "sample_0": [30, 35, 40, 45],
            "sample_1": [31, 36, 41, 46],
            "sample_2": [32, 37, 42, 47],
            "rainfall": [70, 80, 75, 85],
        }
    )

    num_predictions = 1
    updated_historic = extended_predictor.update_historic_data(historic_df, predictions_df, num_predictions)

    assert len(updated_historic) == 6, f"Expected 6 rows, got {len(updated_historic)}"

    loc_a_rows = updated_historic[updated_historic["location"] == "loc_A"]
    loc_b_rows = updated_historic[updated_historic["location"] == "loc_B"]
    assert len(loc_a_rows) == 3
    assert len(loc_b_rows) == 3

    assert "2020-03" in loc_a_rows["time_period"].values
    assert "2020-03" in loc_b_rows["time_period"].values


def test_multi_location_all_locations_in_output():
    """Verify all locations receive predictions in the output."""
    data = create_multi_location_data(num_locations=4)
    mock_model = TrackingMockModel(
        locations=data["locations"],
        future_periods=data["future_periods"],
    )
    extended_predictor = ExtendedPredictor(mock_model, desired_scope=6)

    result = extended_predictor.predict(data["historic_data"], data["future_data"])
    result_df = result.to_pandas()

    result_locations = set(result_df["location"].unique())
    expected_locations = set(data["locations"])

    assert result_locations == expected_locations


def test_multi_location_correct_prediction_count():
    """Each location should have exactly desired_scope predictions."""
    desired_scope = 6
    data = create_multi_location_data(num_locations=4)
    mock_model = TrackingMockModel(
        locations=data["locations"],
        future_periods=data["future_periods"],
    )
    extended_predictor = ExtendedPredictor(mock_model, desired_scope=desired_scope)

    result = extended_predictor.predict(data["historic_data"], data["future_data"])
    result_df = result.to_pandas()

    for loc in data["locations"]:
        loc_df = result_df[result_df["location"] == loc]
        assert len(loc_df) == desired_scope, f"Location {loc} has {len(loc_df)} predictions, expected {desired_scope}"


def test_multi_location_correct_time_periods():
    """Each location should have predictions for the first desired_scope time periods."""
    desired_scope = 6
    data = create_multi_location_data(num_locations=4, num_future_periods=8)
    mock_model = TrackingMockModel(
        locations=data["locations"],
        future_periods=data["future_periods"],
    )
    extended_predictor = ExtendedPredictor(mock_model, desired_scope=desired_scope)

    result = extended_predictor.predict(data["historic_data"], data["future_data"])
    result_df = result.to_pandas()

    expected_periods = set(data["future_periods"][:desired_scope])

    for loc in data["locations"]:
        loc_df = result_df[result_df["location"] == loc]
        loc_periods = set(str(p) for p in loc_df["time_period"].tolist())
        assert loc_periods == expected_periods, f"Location {loc} has periods {loc_periods}, expected {expected_periods}"


def test_multi_location_requires_multiple_iterations():
    """Verify multiple iterations are needed when desired_scope > max_pred_length."""
    desired_scope = 6
    max_pred_length = 3
    data = create_multi_location_data(num_locations=4)
    mock_model = TrackingMockModel(
        locations=data["locations"],
        future_periods=data["future_periods"],
        max_pred_length=max_pred_length,
    )
    extended_predictor = ExtendedPredictor(mock_model, desired_scope=desired_scope)

    extended_predictor.predict(data["historic_data"], data["future_data"])

    assert len(mock_model.call_history) >= 2, (
        f"Expected at least 2 prediction calls, got {len(mock_model.call_history)}"
    )


def test_multi_location_each_iteration_includes_all_locations():
    """Each iteration should receive data for all locations."""
    desired_scope = 6
    data = create_multi_location_data(num_locations=4)
    mock_model = TrackingMockModel(
        locations=data["locations"],
        future_periods=data["future_periods"],
    )
    extended_predictor = ExtendedPredictor(mock_model, desired_scope=desired_scope)

    extended_predictor.predict(data["historic_data"], data["future_data"])

    expected_locations = set(data["locations"])
    for i, call in enumerate(mock_model.call_history):
        assert call["historic_locations"] == expected_locations, f"Call {i}: historic missing locations"
        assert call["future_locations"] == expected_locations, f"Call {i}: future missing locations"


def test_multi_location_historic_data_grows_between_iterations():
    """Historic data should grow for all locations between iterations."""
    desired_scope = 6
    data = create_multi_location_data(num_locations=4)
    mock_model = TrackingMockModel(
        locations=data["locations"],
        future_periods=data["future_periods"],
    )
    extended_predictor = ExtendedPredictor(mock_model, desired_scope=desired_scope)

    extended_predictor.predict(data["historic_data"], data["future_data"])

    for i in range(1, len(mock_model.call_history)):
        prev_call = mock_model.call_history[i - 1]
        curr_call = mock_model.call_history[i]
        for loc in data["locations"]:
            prev_periods = len(prev_call["historic_periods_per_loc"][loc])
            curr_periods = len(curr_call["historic_periods_per_loc"][loc])
            assert curr_periods > prev_periods, f"Call {i}: Historic data for {loc} did not grow"


def test_multi_location_no_data_mixing_between_locations():
    """Verify predictions have correct encoded values (no location/period mixing)."""
    desired_scope = 6
    data = create_multi_location_data(num_locations=4)
    mock_model = TrackingMockModel(
        locations=data["locations"],
        future_periods=data["future_periods"],
    )
    extended_predictor = ExtendedPredictor(mock_model, desired_scope=desired_scope)

    result = extended_predictor.predict(data["historic_data"], data["future_data"])
    result_df = result.to_pandas()

    for loc in data["locations"]:
        loc_idx = data["locations"].index(loc)
        loc_df = result_df[result_df["location"] == loc].sort_values("time_period")
        for _, row in loc_df.iterrows():
            period_str = str(row["time_period"])
            period_idx = data["future_periods"].index(period_str)
            sample_val = row["sample_0"]
            decoded_loc_idx = int(sample_val % 10000) // 100
            decoded_period_idx = int(sample_val % 100)
            assert decoded_loc_idx == loc_idx, f"Location mismatch: expected {loc_idx}, got {decoded_loc_idx}"
            assert decoded_period_idx == period_idx, (
                f"Period mismatch for {loc}: expected {period_idx}, got {decoded_period_idx}"
            )


def test_single_iteration_when_scope_within_max():
    """No multiple iterations needed when desired_scope <= max_pred_length."""
    desired_scope = 3
    max_pred_length = 4
    data = create_multi_location_data(num_locations=2, num_future_periods=6)
    mock_model = TrackingMockModel(
        locations=data["locations"],
        future_periods=data["future_periods"],
        max_pred_length=max_pred_length,
    )
    extended_predictor = ExtendedPredictor(mock_model, desired_scope=desired_scope)

    result = extended_predictor.predict(data["historic_data"], data["future_data"])
    result_df = result.to_pandas()

    assert len(mock_model.call_history) == 1, "Should only need 1 iteration"

    for loc in data["locations"]:
        loc_df = result_df[result_df["location"] == loc]
        assert len(loc_df) == desired_scope


def test_single_location_prediction():
    """Verify single location predictions still work correctly."""
    locations = ["single_loc"]
    future_periods = ["2020-11", "2020-12", "2021-01", "2021-02", "2021-03", "2021-04"]

    historic_rows = [
        {"time_period": f"2020-{m:02d}", "location": "single_loc", "disease_cases": 100, "rainfall": 50.0}
        for m in range(1, 11)
    ]
    historic_data = DataSet.from_pandas(pd.DataFrame(historic_rows))  # type: ignore[reportArgumentType]

    future_rows = [{"time_period": p, "location": "single_loc", "rainfall": 60.0} for p in future_periods]
    future_data = DataSet.from_pandas(pd.DataFrame(future_rows))  # type: ignore[reportArgumentType]

    mock_model = TrackingMockModel(
        locations=locations,
        future_periods=future_periods,
        max_pred_length=3,
    )
    extended_predictor = ExtendedPredictor(mock_model, desired_scope=5)

    result = extended_predictor.predict(historic_data, future_data)
    result_df = result.to_pandas()

    assert len(result_df) == 5
    assert set(result_df["location"].unique()) == {"single_loc"}
    assert len(result_df["time_period"].unique()) == 5


def test_update_historic_data_averages_samples():
    """Verify disease_cases is computed as the mean of sample columns."""
    mock_model = MockModel()
    extended_predictor = ExtendedPredictor(mock_model, 6)

    historic_df = pd.DataFrame(
        {
            "time_period": ["2020-01"],
            "location": ["loc_A"],
            "disease_cases": [10],
            "rainfall": [50],
        }
    )

    predictions_df = pd.DataFrame(
        {
            "time_period": ["2020-02"],
            "location": ["loc_A"],
            "sample_0": [100.0],
            "sample_1": [200.0],
            "sample_2": [300.0],
            "rainfall": [60],
        }
    )

    updated_historic = extended_predictor.update_historic_data(historic_df, predictions_df, num_predictions=1)

    new_row = updated_historic[updated_historic["time_period"] == "2020-02"].iloc[0]
    expected_mean = (100.0 + 200.0 + 300.0) / 3
    assert new_row["disease_cases"] == expected_mean, (
        f"Expected disease_cases={expected_mean}, got {new_row['disease_cases']}"
    )
    assert "sample_0" not in updated_historic.columns


def test_overlap_handling_keeps_later_prediction():
    """Test that overlapping predictions keep the later (shorter horizon) prediction.

    With min=2, max=3, scope=5:
    - Iteration 1 predicts periods [0, 1, 2]
    - Iteration 2 predicts periods [2, 3, 4]
    - Period 2 is predicted twice; should keep iteration 2's prediction.
    """
    desired_scope = 5
    min_pred_length = 2
    max_pred_length = 3
    data = create_multi_location_data(num_locations=2, num_future_periods=6)

    mock_model = TrackingMockModel(
        locations=data["locations"],
        future_periods=data["future_periods"],
        min_pred_length=min_pred_length,
        max_pred_length=max_pred_length,
    )
    extended_predictor = ExtendedPredictor(mock_model, desired_scope=desired_scope)

    result = extended_predictor.predict(data["historic_data"], data["future_data"])
    result_df = result.to_pandas()

    assert len(mock_model.call_history) == 2, "Should have exactly 2 iterations"

    # Verify iteration 1 predicted periods 0, 1, 2
    iter1_periods = [str(p) for p in mock_model.call_history[0]["future_periods_per_loc"][data["locations"][0]]]
    assert iter1_periods == data["future_periods"][:3], f"Iteration 1 should predict periods 0-2, got {iter1_periods}"

    # Verify iteration 2 predicted periods 2, 3, 4
    iter2_periods = [str(p) for p in mock_model.call_history[1]["future_periods_per_loc"][data["locations"][0]]]
    assert iter2_periods == data["future_periods"][2:5], f"Iteration 2 should predict periods 2-4, got {iter2_periods}"

    # Check that period 2 has the value from iteration 2 (call_num=1), not iteration 1
    # Encoding: call_num * 10000 + loc_idx * 100 + period_idx
    # Period 2 from iteration 2: 1 * 10000 + loc_idx * 100 + 2 = 10000 + loc_idx * 100 + 2
    for loc in data["locations"]:
        loc_idx = data["locations"].index(loc)
        loc_df = result_df[result_df["location"] == loc]
        period_2_row = loc_df[loc_df["time_period"].astype(str) == data["future_periods"][2]].iloc[0]
        sample_val = period_2_row["sample_0"]

        decoded_call_num = int(sample_val) // 10000
        assert decoded_call_num == 1, (
            f"Period 2 for {loc} should be from iteration 2 (call_num=1), but got call_num={decoded_call_num}"
        )


def test_overlap_correct_final_count():
    """Test that overlapping predictions result in correct final count after deduplication."""
    desired_scope = 5
    data = create_multi_location_data(num_locations=3, num_future_periods=6)

    mock_model = TrackingMockModel(
        locations=data["locations"],
        future_periods=data["future_periods"],
        min_pred_length=2,
        max_pred_length=3,
    )
    extended_predictor = ExtendedPredictor(mock_model, desired_scope=desired_scope)

    result = extended_predictor.predict(data["historic_data"], data["future_data"])
    result_df = result.to_pandas()

    # Should have exactly desired_scope predictions per location after deduplication
    for loc in data["locations"]:
        loc_df = result_df[result_df["location"] == loc]
        assert len(loc_df) == desired_scope, (
            f"Location {loc} should have {desired_scope} predictions after deduplication, got {len(loc_df)}"
        )
