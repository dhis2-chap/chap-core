"""Tests for the evaluation abstraction module."""

import pandas as pd
import pytest

from chap_core.assessment.evaluation import Evaluation, EvaluationBase, FlatEvaluationData
from chap_core.assessment.flat_representations import (
    FlatForecasts,
    FlatObserved,
    convert_backtest_observations_to_flat_observations,
    convert_backtest_to_flat_forecasts,
)


class TestFlatEvaluationData:
    """Tests for FlatEvaluationData dataclass."""

    def test_flat_evaluation_data_creation(self, flat_forecasts, flat_observations):
        """Test that FlatEvaluationData can be created with forecasts and observations."""
        flat_data = FlatEvaluationData(forecasts=flat_forecasts, observations=flat_observations)

        assert flat_data.forecasts is flat_forecasts
        assert flat_data.observations is flat_observations

    def test_flat_evaluation_data_is_dataclass(self):
        """Test that FlatEvaluationData is a dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(FlatEvaluationData)


class TestEvaluationBase:
    """Tests for EvaluationBase ABC."""

    def test_evaluation_base_is_abstract(self):
        """Test that EvaluationBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EvaluationBase()

    def test_evaluation_base_requires_abstract_methods(self):
        """Test that EvaluationBase subclasses must implement abstract methods."""

        # Create a subclass that doesn't implement abstract methods
        class IncompleteEvaluation(EvaluationBase):
            pass

        with pytest.raises(TypeError):
            IncompleteEvaluation()


class TestEvaluation:
    """Tests for Evaluation concrete implementation."""

    def test_from_backtest_creates_evaluation(self, backtest):
        """Test that from_backtest creates an Evaluation instance."""
        evaluation = Evaluation.from_backtest(backtest)

        assert isinstance(evaluation, Evaluation)
        assert isinstance(evaluation, EvaluationBase)

    def test_to_backtest_returns_wrapped_object(self, backtest):
        """Test that to_backtest returns the original BackTest object."""
        evaluation = Evaluation.from_backtest(backtest)

        result = evaluation.to_backtest()

        assert result is backtest

    def test_get_org_units_returns_correct_data(self, backtest):
        """Test that get_org_units returns the org_units from BackTest."""
        evaluation = Evaluation.from_backtest(backtest)

        org_units = evaluation.get_org_units()

        assert org_units == backtest.org_units

    def test_get_split_periods_returns_correct_data(self, backtest):
        """Test that get_split_periods returns the split_periods from BackTest."""
        evaluation = Evaluation.from_backtest(backtest)

        split_periods = evaluation.get_split_periods()

        assert split_periods == backtest.split_periods

    def test_to_flat_returns_flat_evaluation_data(self, backtest):
        """Test that to_flat returns a FlatEvaluationData instance."""
        evaluation = Evaluation.from_backtest(backtest)

        flat_data = evaluation.to_flat()

        assert isinstance(flat_data, FlatEvaluationData)
        assert isinstance(flat_data.forecasts, pd.DataFrame)
        assert isinstance(flat_data.observations, pd.DataFrame)

    def test_to_flat_forecasts_match_conversion_function(self, backtest):
        """Test that to_flat forecasts match the existing conversion function."""
        evaluation = Evaluation.from_backtest(backtest)

        flat_data = evaluation.to_flat()
        expected_forecasts_df = convert_backtest_to_flat_forecasts(backtest.forecasts)

        pd.testing.assert_frame_equal(flat_data.forecasts, expected_forecasts_df, check_dtype=False)

    def test_to_flat_observations_match_conversion_function(self, backtest):
        """Test that to_flat observations match the existing conversion function."""
        evaluation = Evaluation.from_backtest(backtest)

        flat_data = evaluation.to_flat()
        expected_observations_df = convert_backtest_observations_to_flat_observations(backtest.dataset.observations)

        pd.testing.assert_frame_equal(flat_data.observations, expected_observations_df, check_dtype=False)

    def test_to_flat_caches_result(self, backtest):
        """Test that to_flat caches the result and returns the same object."""
        evaluation = Evaluation.from_backtest(backtest)

        flat_data_1 = evaluation.to_flat()
        flat_data_2 = evaluation.to_flat()

        # Should return the exact same object (not just equal data)
        assert flat_data_1 is flat_data_2

    def test_to_flat_with_empty_forecasts(self, backtest_empty):
        """Test that to_flat works with empty forecasts."""
        evaluation = Evaluation.from_backtest(backtest_empty)
        flat_data = evaluation.to_flat()

        assert isinstance(flat_data, FlatEvaluationData)
        assert len(flat_data.forecasts) == 0

    def test_to_flat_preserves_forecast_structure(self, backtest):
        """Test that to_flat preserves the forecast data structure."""
        evaluation = Evaluation.from_backtest(backtest)

        flat_data = evaluation.to_flat()

        # Check that required columns exist
        required_cols = ["location", "time_period", "horizon_distance", "sample", "forecast"]
        for col in required_cols:
            assert col in flat_data.forecasts.columns

    def test_to_flat_preserves_observation_structure(self, backtest):
        """Test that to_flat preserves the observation data structure."""
        evaluation = Evaluation.from_backtest(backtest)

        flat_data = evaluation.to_flat()

        # Check that required columns exist
        required_cols = ["location", "time_period", "disease_cases"]
        for col in required_cols:
            assert col in flat_data.observations.columns
