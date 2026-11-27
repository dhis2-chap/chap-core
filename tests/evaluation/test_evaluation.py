"""Tests for the evaluation abstraction module."""

import pandas as pd

from chap_core.assessment.evaluation import Evaluation, ModelCard
from chap_core.assessment.flat_representations import (
    convert_backtest_observations_to_flat_observations,
    convert_backtest_to_flat_forecasts,
)


class TestEvaluation:
    """Tests for Evaluation concrete implementation."""

    def test_to_backtest_returns_wrapped_object(self, backtest):
        """Test that to_backtest returns the original Backtest object."""
        evaluation = Evaluation.from_backtest(backtest)

        result = evaluation.to_backtest()

        assert result is backtest

    def test_get_org_units_returns_correct_data(self, backtest):
        """Test that get_org_units returns the org_units from Backtest."""
        evaluation = Evaluation.from_backtest(backtest)

        org_units = evaluation.get_org_units()

        assert org_units == backtest.org_units

    def test_get_split_periods_returns_correct_data(self, backtest):
        """Test that get_split_periods returns the split_periods from Backtest."""
        evaluation = Evaluation.from_backtest(backtest)

        split_periods = evaluation.get_split_periods()

        assert split_periods == backtest.split_periods

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

    def test_to_backtest_records_the_storage_identity(self):
        """The configured model, specification and name given to to_backtest end up on the row,
        and a persisted backtest reaches its observations through its dataset, not an embedded one."""
        from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB, ModelTemplateDB
        from chap_core.database.tables import BacktestSpecification

        template = ModelTemplateDB(id=3, name="template", version="2.0")
        configured_model = ConfiguredModelDB(id=5, name="template:conf", model_template_id=3, model_template=template)
        specification = BacktestSpecification(dataset_id=7)

        backtest = self._evaluation().to_backtest(configured_model, specification, "rest")

        assert (backtest.name, backtest.dataset_id, backtest.model_id) == ("rest", 7, "template:conf")
        assert (backtest.model_db_id, backtest.model_template_version) == (5, "2.0")
        assert backtest.specification is specification
        assert backtest.forecasts
        assert backtest.dataset is None

    def test_to_backtest_identity_cannot_change_once_built(self):
        import pytest

        evaluation = self._evaluation()
        evaluation.to_backtest()

        with pytest.raises(ValueError):
            evaluation.to_backtest(name="other")

    def test_with_model_round_trips_through_a_file(self, tmp_path):
        from chap_core.assessment.evaluation_metadata import EvaluatedModel

        model = EvaluatedModel(name="m", version="1.2", configuration={"user_option_values": {"lags": 3}})
        path = tmp_path / "eval.nc"
        self._evaluation().with_model(model).to_file(path)

        assert Evaluation.from_file(path).metadata.model == model

    @staticmethod
    def _evaluation() -> Evaluation:
        import numpy as np

        from chap_core.datatypes import SamplesWithTruth
        from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
        from chap_core.time_period import PeriodRange

        periods = PeriodRange.from_strings(["2022-01", "2022-02"])
        result = DataSet(
            {"A": SamplesWithTruth(time_period=periods, samples=np.ones((2, 3)), disease_cases=np.ones(2))}
        )
        return Evaluation.from_samples_with_truth([result], periods[0])

    def test_create_carries_dataset_population(self, health_population_data):
        """The CLI path keeps the dataset's population, so incidence metrics like org unit ranking apply."""
        from chap_core.api_types import BacktestParams
        from chap_core.assessment.metrics import OrgUnitRankingMetric
        from chap_core.predictor.naive_estimator import NaiveEstimator

        evaluation = Evaluation.create(
            estimator=NaiveEstimator(),
            dataset=health_population_data,
            backtest_params=BacktestParams(n_periods=2, n_splits=2, stride=1),
        )

        observations = evaluation.to_flat().observations
        assert observations["population"].notna().all()
        assert OrgUnitRankingMetric().is_applicable(observations)

    def test_from_samples_with_truth_skips_a_location_without_rows(self):
        """A location with no forecast rows gets no forecasts instead of crashing the split."""
        import numpy as np

        from chap_core.datatypes import SamplesWithTruth
        from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
        from chap_core.time_period import PeriodRange

        periods = PeriodRange.from_strings(["2022-01", "2022-02"])
        result = DataSet(
            {
                "A": SamplesWithTruth(time_period=periods, samples=np.ones((2, 3)), disease_cases=np.ones(2)),
                "B": SamplesWithTruth(time_period=periods[:0], samples=np.ones((0, 3)), disease_cases=np.ones(0)),
            }
        )

        evaluation = Evaluation.from_samples_with_truth([result], periods[0])

        assert set(pd.DataFrame(evaluation.to_flat().forecasts)["location"]) == {"A"}

    def test_to_backtest_drops_nan_samples(self):
        """NaN samples are left out of a forecast, and a forecast with only NaN samples gets no row."""
        import numpy as np

        from chap_core.datatypes import SamplesWithTruth
        from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
        from chap_core.time_period import PeriodRange

        periods = PeriodRange.from_strings(["2022-01", "2022-02"])
        samples = np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, np.nan]])
        result = DataSet({"A": SamplesWithTruth(time_period=periods, samples=samples, disease_cases=np.ones(2))})
        evaluation = Evaluation.from_samples_with_truth([result], periods[0])

        forecasts = evaluation.to_backtest().forecasts

        assert [(f.org_unit, f.period, f.values) for f in forecasts] == [("A", "202201", [1.0, 3.0])]

    class TestModelCard:
        """Tests for ModelCard class"""

        def test_modelcard_inherits_backtest(self, backtest):
            """Test to assert that ModelCard wraps backtest"""
            modelCard = ModelCard(backtest)
            assert modelCard.backtest is backtest

        def test_modelcard_description(self, backtest):
            """Test that description is given"""
            description = "Test description"
            modelCard = ModelCard(backtest, description)
            assert modelCard.description == description
