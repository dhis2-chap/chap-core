"""Tests for the preference_learn CLI endpoint."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chap_core.api_types import BackTestParams, RunConfig
from chap_core.cli_endpoints.preference_learn import (
    PreferenceLearningParams,
    preference_learn,
    _compute_metrics,
    _create_evaluation,
)
from chap_core.preference_learning.preference_learner import (
    ModelCandidate,
    TournamentPreferenceLearner,
)


class TestPreferenceLearningParams:
    def test_default_values(self):
        params = PreferenceLearningParams()
        assert params.max_iterations == 10
        assert params.decision_mode == "visual"
        assert params.decision_metrics == ["mae"]
        assert params.lower_is_better is True

    def test_custom_values(self):
        params = PreferenceLearningParams(
            max_iterations=5,
            decision_mode="metric",
            decision_metrics=["rmse", "mae"],
            lower_is_better=False,
        )
        assert params.max_iterations == 5
        assert params.decision_mode == "metric"
        assert params.decision_metrics == ["rmse", "mae"]
        assert params.lower_is_better is False


class TestPreferenceLearnCLI:
    @patch("chap_core.cli_endpoints.preference_learn._create_evaluation")
    @patch("chap_core.cli_endpoints.preference_learn._compute_metrics")
    @patch("chap_core.cli_endpoints.preference_learn.load_dataset_from_csv")
    @patch("chap_core.cli_endpoints.preference_learn.discover_geojson")
    def test_preference_learn_metric_mode(
        self,
        mock_discover_geojson,
        mock_load_dataset,
        mock_compute_metrics,
        mock_create_evaluation,
        tmp_path,
    ):
        """Test preference learning with metric-based decision making."""
        # Setup mocks
        mock_discover_geojson.return_value = None
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset

        # Mock evaluations
        mock_eval1 = MagicMock()
        mock_eval2 = MagicMock()
        mock_create_evaluation.side_effect = [mock_eval1, mock_eval2]

        # Mock metrics - second model is better (lower MAE)
        mock_compute_metrics.side_effect = [
            {"mae": 0.8, "rmse": 1.0},
            {"mae": 0.5, "rmse": 0.7},
        ]

        # Create test files
        dataset_csv = tmp_path / "test_data.csv"
        dataset_csv.write_text("location,time_period,disease_cases\nA,2020-01,10")

        search_space_yaml = tmp_path / "search_space.yaml"
        search_space_yaml.write_text(
            """
learning_rate:
  values: [0.01, 0.1]
"""
        )

        state_file = tmp_path / "state.json"

        # Run preference learning with max 1 iteration
        learning_params = PreferenceLearningParams(
            max_iterations=1,
            decision_mode="metric",
            decision_metrics=["mae"],
        )

        preference_learn(
            model_name="test_model",
            dataset_csv=dataset_csv,
            search_space_yaml=search_space_yaml,
            state_file=state_file,
            backtest_params=BackTestParams(n_periods=2, n_splits=2, stride=1),
            run_config=RunConfig(),
            learning_params=learning_params,
        )

        # Verify state was saved
        assert state_file.exists()

        # Load and verify state
        learner = TournamentPreferenceLearner.load(state_file)
        assert learner.current_iteration == 1

        # Verify best candidate was selected (second one with lower MAE)
        best = learner.get_best_candidate()
        assert best is not None
        assert best.configuration["learning_rate"] == 0.1

    @patch("chap_core.cli_endpoints.preference_learn._create_evaluation")
    @patch("chap_core.cli_endpoints.preference_learn._compute_metrics")
    @patch("chap_core.cli_endpoints.preference_learn.load_dataset_from_csv")
    @patch("chap_core.cli_endpoints.preference_learn.discover_geojson")
    def test_preference_learn_resumes_from_state(
        self,
        mock_discover_geojson,
        mock_load_dataset,
        mock_compute_metrics,
        mock_create_evaluation,
        tmp_path,
    ):
        """Test that preference learning resumes from existing state file."""
        # Setup mocks
        mock_discover_geojson.return_value = None
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset
        mock_create_evaluation.return_value = MagicMock()
        mock_compute_metrics.return_value = {"mae": 0.5}

        # Create initial state with 1 iteration already done
        search_space = {"param": [1, 2, 3]}
        learner = TournamentPreferenceLearner.init(
            model_name="test_model",
            search_space=search_space,
            max_iterations=3,
        )
        candidates = learner.get_next_candidates()
        learner.report_preference(candidates, 0, [{"mae": 0.5}, {"mae": 0.6}])

        state_file = tmp_path / "state.json"
        learner.save(state_file)

        # Create test files
        dataset_csv = tmp_path / "test_data.csv"
        dataset_csv.write_text("location,time_period,disease_cases\nA,2020-01,10")

        # Run preference learning - should resume from iteration 1
        learning_params = PreferenceLearningParams(
            max_iterations=2,  # Will stop after 2 total iterations
            decision_mode="metric",
        )

        preference_learn(
            model_name="test_model",
            dataset_csv=dataset_csv,
            state_file=state_file,
            backtest_params=BackTestParams(n_periods=2, n_splits=2, stride=1),
            run_config=RunConfig(),
            learning_params=learning_params,
        )

        # Verify we continued from existing state
        resumed_learner = TournamentPreferenceLearner.load(state_file)
        assert resumed_learner.current_iteration == 2
        assert len(resumed_learner.get_comparison_history()) == 2

    def test_preference_learn_no_search_space_raises(self, tmp_path):
        """Test that missing search space raises appropriate error."""
        dataset_csv = tmp_path / "test_data.csv"
        dataset_csv.write_text("location,time_period,disease_cases\nA,2020-01,10")

        with patch("chap_core.cli_endpoints.preference_learn.discover_geojson") as mock_geojson:
            with patch("chap_core.cli_endpoints.preference_learn.load_dataset_from_csv") as mock_load:
                with patch("chap_core.cli_endpoints.preference_learn.ModelTemplate") as mock_template:
                    mock_geojson.return_value = None
                    mock_load.return_value = MagicMock()

                    # Mock template with no search space
                    mock_template_instance = MagicMock()
                    mock_template_instance.model_template_config.hpo_search_space = None
                    mock_template.from_directory_or_github_url.return_value = mock_template_instance

                    with pytest.raises(ValueError, match="No search space provided"):
                        preference_learn(
                            model_name="test_model",
                            dataset_csv=dataset_csv,
                            state_file=tmp_path / "state.json",
                        )
