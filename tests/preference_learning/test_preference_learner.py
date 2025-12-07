"""Tests for PreferenceLearner."""

from pathlib import Path

import pytest

from chap_core.preference_learning.preference_learner import (
    ComparisonResult,
    ModelCandidate,
    PreferenceLearnerBase,
    PreferenceLearnerState,
    TournamentPreferenceLearner,
)


class TestModelCandidate:
    def test_create_model_candidate(self):
        candidate = ModelCandidate(model_name="naive_model")
        assert candidate.model_name == "naive_model"
        assert candidate.configuration == {}

    def test_model_candidate_with_config(self):
        candidate = ModelCandidate(
            model_name="custom_model",
            configuration={"learning_rate": 0.01},
        )
        assert candidate.model_name == "custom_model"
        assert candidate.configuration["learning_rate"] == 0.01

    def test_model_candidate_hash(self):
        c1 = ModelCandidate(model_name="model_a")
        c2 = ModelCandidate(model_name="model_a")
        c3 = ModelCandidate(model_name="model_b")

        assert hash(c1) == hash(c2)
        assert hash(c1) != hash(c3)

    def test_model_candidate_equality(self):
        c1 = ModelCandidate(model_name="model_a", configuration={"x": 1})
        c2 = ModelCandidate(model_name="model_a", configuration={"x": 1})
        c3 = ModelCandidate(model_name="model_a", configuration={"x": 2})

        assert c1 == c2
        assert c1 != c3


class TestComparisonResult:
    def test_preferred_property(self):
        c1 = ModelCandidate(model_name="model_a")
        c2 = ModelCandidate(model_name="model_b")
        result = ComparisonResult(
            candidates=[c1, c2],
            preferred_index=1,
            metrics=[{"mae": 0.8}, {"mae": 0.5}],
            iteration=0,
        )
        assert result.preferred == c2
        assert result.preferred.model_name == "model_b"


class TestPreferenceLearnerState:
    def test_state_to_dict(self):
        candidate = ModelCandidate(model_name="test_model")
        state = PreferenceLearnerState(
            candidates=[candidate],
            current_iteration=1,
            best_candidate=candidate,
        )
        data = state.to_dict()

        assert len(data["candidates"]) == 1
        assert data["current_iteration"] == 1
        assert data["best_candidate"]["model_name"] == "test_model"

    def test_state_from_dict(self):
        data = {
            "candidates": [{"model_name": "model_a", "configuration": {}}],
            "comparison_history": [],
            "current_iteration": 5,
            "best_candidate": {"model_name": "model_a", "configuration": {}},
        }
        state = PreferenceLearnerState.from_dict(data)

        assert len(state.candidates) == 1
        assert state.current_iteration == 5
        assert state.best_candidate.model_name == "model_a"

    def test_state_roundtrip(self):
        c1 = ModelCandidate(model_name="model_a")
        c2 = ModelCandidate(model_name="model_b")
        result = ComparisonResult(
            candidates=[c1, c2],
            preferred_index=0,
            metrics=[{"mae": 0.5}, {"mae": 0.7}],
            iteration=0,
        )
        state = PreferenceLearnerState(
            candidates=[c1, c2],
            comparison_history=[result],
            current_iteration=1,
            best_candidate=c1,
        )

        data = state.to_dict()
        restored = PreferenceLearnerState.from_dict(data)

        assert len(restored.candidates) == 2
        assert len(restored.comparison_history) == 1
        assert restored.comparison_history[0].preferred.model_name == "model_a"
        assert restored.comparison_history[0].preferred_index == 0


class TestTournamentPreferenceLearner:
    def test_create_learner(self):
        candidates = [
            ModelCandidate(model_name="model_a"),
            ModelCandidate(model_name="model_b"),
        ]
        learner = TournamentPreferenceLearner(candidates=candidates)

        assert learner.current_iteration == 0
        assert not learner.is_complete()

    def test_implements_abc(self):
        candidates = [
            ModelCandidate(model_name="model_a"),
            ModelCandidate(model_name="model_b"),
        ]
        learner = TournamentPreferenceLearner(candidates=candidates)
        assert isinstance(learner, PreferenceLearnerBase)

    def test_get_next_candidates(self):
        candidates = [
            ModelCandidate(model_name="model_a"),
            ModelCandidate(model_name="model_b"),
        ]
        learner = TournamentPreferenceLearner(candidates=candidates)

        next_candidates = learner.get_next_candidates()
        assert next_candidates is not None
        assert len(next_candidates) == 2

    def test_get_next_candidates_not_enough(self):
        candidates = [ModelCandidate(model_name="model_a")]
        learner = TournamentPreferenceLearner(candidates=candidates)

        next_candidates = learner.get_next_candidates()
        assert next_candidates is None

    def test_report_preference(self):
        candidates = [
            ModelCandidate(model_name="model_a"),
            ModelCandidate(model_name="model_b"),
        ]
        learner = TournamentPreferenceLearner(candidates=candidates)

        next_candidates = learner.get_next_candidates()
        learner.report_preference(
            candidates=next_candidates,
            preferred_index=0,
            metrics=[{"mae": 0.5}, {"mae": 0.7}],
        )

        assert learner.current_iteration == 1
        assert learner.get_best_candidate() == next_candidates[0]

    def test_max_iterations(self):
        candidates = [
            ModelCandidate(model_name="model_a"),
            ModelCandidate(model_name="model_b"),
        ]
        learner = TournamentPreferenceLearner(candidates=candidates, max_iterations=2)

        # Report two preferences
        for _ in range(2):
            next_candidates = learner.get_next_candidates()
            if next_candidates:
                learner.report_preference(
                    candidates=next_candidates,
                    preferred_index=0,
                    metrics=[{"mae": 0.5}, {"mae": 0.7}],
                )

        assert learner.is_complete()

    def test_persistence(self, tmp_path):
        state_file = tmp_path / "state.json"
        candidates = [
            ModelCandidate(model_name="model_a"),
            ModelCandidate(model_name="model_b"),
        ]

        # Create learner and report a preference
        learner = TournamentPreferenceLearner(candidates=candidates, state_file=state_file)
        next_candidates = learner.get_next_candidates()
        learner.report_preference(
            candidates=next_candidates,
            preferred_index=0,
            metrics=[{"mae": 0.5}, {"mae": 0.7}],
        )

        assert state_file.exists()

        # Create new learner from state file
        learner2 = TournamentPreferenceLearner(candidates=candidates, state_file=state_file)
        assert learner2.current_iteration == 1
        assert len(learner2.get_comparison_history()) == 1

    def test_comparison_history(self):
        candidates = [
            ModelCandidate(model_name="model_a"),
            ModelCandidate(model_name="model_b"),
        ]
        learner = TournamentPreferenceLearner(candidates=candidates)

        next_candidates = learner.get_next_candidates()
        learner.report_preference(
            candidates=next_candidates,
            preferred_index=1,
            metrics=[{"mae": 0.8, "rmse": 1.0}, {"mae": 0.5, "rmse": 0.7}],
        )

        history = learner.get_comparison_history()
        assert len(history) == 1
        assert history[0].preferred.model_name == "model_b"
        assert history[0].metrics[0]["mae"] == 0.8
        assert history[0].metrics[1]["mae"] == 0.5
