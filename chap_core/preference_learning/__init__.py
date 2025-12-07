"""Preference learning module for model selection via A/B testing."""

from chap_core.preference_learning.preference_learner import PreferenceLearner
from chap_core.preference_learning.decision_maker import DecisionMaker, MetricBasedDecisionMaker

__all__ = ["PreferenceLearner", "DecisionMaker", "MetricBasedDecisionMaker"]
