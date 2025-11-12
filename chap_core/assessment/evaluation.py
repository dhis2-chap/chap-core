"""
Evaluation abstraction for model evaluation results.

This module provides a database-agnostic interface for working with model
evaluation results, enabling better code reuse between REST API and CLI workflows.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from chap_core.assessment.flat_representations import (
    FlatForecasts,
    FlatObserved,
    convert_backtest_observations_to_flat_observations,
    convert_backtest_to_flat_forecasts,
)

if TYPE_CHECKING:
    from chap_core.database.tables import BackTest


@dataclass
class FlatEvaluationData:
    """
    Container for flat representations of evaluation data.

    Combines forecasts and observations which are always used together
    for metric computation and visualization.

    Attributes:
        forecasts: Flat representation of forecast samples
        observations: Flat representation of observed values
    """

    forecasts: FlatForecasts
    observations: FlatObserved


class EvaluationBase(ABC):
    """
    Abstract base class for evaluation results.

    An Evaluation represents the complete results of evaluating a model:
    - Forecasts (with samples/quantiles)
    - Observations (ground truth)
    - Metadata (locations, split periods)

    This abstraction is database-agnostic and can be implemented by
    different concrete classes (database-backed, in-memory, etc.).
    """

    @abstractmethod
    def to_flat(self) -> FlatEvaluationData:
        """
        Export evaluation data as flat representations.

        Returns:
            FlatEvaluationData containing FlatForecasts and FlatObserved objects
        """
        pass

    @abstractmethod
    def get_org_units(self) -> List[str]:
        """
        Get list of locations included in this evaluation.

        Returns:
            List of location identifiers (org_units)
        """
        pass

    @abstractmethod
    def get_split_periods(self) -> List[str]:
        """
        Get list of train/test split periods used in evaluation.

        Returns:
            List of period identifiers (e.g., ["2024-01", "2024-02"])
        """
        pass

    @classmethod
    @abstractmethod
    def from_backtest(cls, backtest: "BackTest") -> "EvaluationBase":
        """
        Create Evaluation from database BackTest object.

        All implementations must support loading from database.

        Args:
            backtest: Database BackTest object (with relationships loaded)

        Returns:
            Evaluation instance
        """
        pass


class Evaluation(EvaluationBase):
    """
    Evaluation implementation backed by database BackTest model.

    This wraps an existing BackTest object and provides the
    EvaluationBase interface without modifying the database schema.
    """

    def __init__(self, backtest: "BackTest"):
        """
        Initialize Evaluation with a BackTest object.

        Args:
            backtest: Database BackTest object (with relationships loaded)
        """
        self._backtest = backtest
        self._flat_data_cache: Optional[FlatEvaluationData] = None

    @classmethod
    def from_backtest(cls, backtest: "BackTest") -> "Evaluation":
        """
        Create Evaluation from database BackTest object.

        Args:
            backtest: Database BackTest object (with relationships loaded)

        Returns:
            Evaluation instance wrapping the BackTest
        """
        return cls(backtest)

    def to_backtest(self) -> "BackTest":
        """
        Get underlying database BackTest object.

        Returns:
            BackTest database model
        """
        return self._backtest

    def to_flat(self) -> FlatEvaluationData:
        """
        Export evaluation data using existing conversion functions.

        Results are cached for performance. Repeated calls return the cached result.

        Returns:
            FlatEvaluationData containing forecasts and observations
        """
        if self._flat_data_cache is None:
            forecasts_df = convert_backtest_to_flat_forecasts(self._backtest.forecasts)
            observations_df = convert_backtest_observations_to_flat_observations(self._backtest.dataset.observations)

            self._flat_data_cache = FlatEvaluationData(
                forecasts=FlatForecasts(forecasts_df),
                observations=FlatObserved(observations_df),
            )
        return self._flat_data_cache

    def get_org_units(self) -> List[str]:
        """
        Get locations from BackTest metadata.

        Returns:
            List of location identifiers
        """
        return self._backtest.org_units

    def get_split_periods(self) -> List[str]:
        """
        Get split periods from BackTest metadata.

        Returns:
            List of period identifiers
        """
        return self._backtest.split_periods
