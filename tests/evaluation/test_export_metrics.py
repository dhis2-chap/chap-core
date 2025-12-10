"""Tests for export_metrics CLI command."""

from pathlib import Path

import pandas as pd
import pytest

from chap_core.assessment.evaluation import Evaluation
from chap_core.cli_endpoints.utils import export_metrics


class TestExportMetrics:
    """Tests for export_metrics function."""

    def test_export_metrics_single_file(self, backtest, tmp_path):
        """Test exporting metrics from a single backtest file."""
        # Create evaluation file
        evaluation = Evaluation.from_backtest(backtest)
        input_file = tmp_path / "eval1.nc"
        evaluation.to_file(
            filepath=input_file,
            model_name="TestModel",
            model_version="1.0.0",
        )

        output_file = tmp_path / "metrics.csv"

        export_metrics(
            input_files=[input_file],
            output_file=output_file,
        )

        assert output_file.exists()

        df = pd.read_csv(output_file)
        assert len(df) == 1
        assert "filename" in df.columns
        assert "model_name" in df.columns
        assert "model_version" in df.columns
        assert df["filename"].iloc[0] == "eval1.nc"
        assert df["model_name"].iloc[0] == "TestModel"
        assert df["model_version"].iloc[0] == "1.0.0"

    def test_export_metrics_multiple_files(self, backtest, tmp_path):
        """Test exporting metrics from multiple backtest files."""
        # Create two evaluation files
        evaluation = Evaluation.from_backtest(backtest)

        input_file1 = tmp_path / "eval1.nc"
        evaluation.to_file(
            filepath=input_file1,
            model_name="Model_A",
            model_version="1.0.0",
        )

        input_file2 = tmp_path / "eval2.nc"
        evaluation.to_file(
            filepath=input_file2,
            model_name="Model_B",
            model_version="2.0.0",
        )

        output_file = tmp_path / "metrics.csv"

        export_metrics(
            input_files=[input_file1, input_file2],
            output_file=output_file,
        )

        df = pd.read_csv(output_file)
        assert len(df) == 2
        assert set(df["filename"]) == {"eval1.nc", "eval2.nc"}
        assert set(df["model_name"]) == {"Model_A", "Model_B"}

    def test_export_metrics_with_specific_metric_ids(self, backtest, tmp_path):
        """Test exporting only specific metrics."""
        evaluation = Evaluation.from_backtest(backtest)
        input_file = tmp_path / "eval.nc"
        evaluation.to_file(filepath=input_file)

        output_file = tmp_path / "metrics.csv"

        export_metrics(
            input_files=[input_file],
            output_file=output_file,
            metric_ids=["crps", "ratio_within_10th_90th"],
        )

        df = pd.read_csv(output_file)
        # Should only have metadata columns plus the two requested metrics
        expected_columns = {"filename", "model_name", "model_version", "crps", "ratio_within_10th_90th"}
        assert set(df.columns) == expected_columns

    def test_export_metrics_invalid_metric_id(self, backtest, tmp_path):
        """Test that invalid metric IDs raise an error."""
        evaluation = Evaluation.from_backtest(backtest)
        input_file = tmp_path / "eval.nc"
        evaluation.to_file(filepath=input_file)

        output_file = tmp_path / "metrics.csv"

        with pytest.raises(ValueError, match="Invalid metric IDs"):
            export_metrics(
                input_files=[input_file],
                output_file=output_file,
                metric_ids=["nonexistent_metric"],
            )

    def test_export_metrics_includes_aggregate_metrics(self, backtest, tmp_path):
        """Test that all aggregate metrics are included by default."""
        evaluation = Evaluation.from_backtest(backtest)
        input_file = tmp_path / "eval.nc"
        evaluation.to_file(filepath=input_file)

        output_file = tmp_path / "metrics.csv"

        export_metrics(
            input_files=[input_file],
            output_file=output_file,
        )

        df = pd.read_csv(output_file)
        # Check that all aggregate metrics are present
        assert "crps" in df.columns
        assert "ratio_within_10th_90th" in df.columns
        assert "ratio_within_25th_75th" in df.columns
        assert "test_sample_count" in df.columns

    def test_export_metrics_with_weekly_data(self, backtest_weeks, tmp_path):
        """Test exporting metrics from weekly backtest data."""
        evaluation = Evaluation.from_backtest(backtest_weeks)
        input_file = tmp_path / "eval_weekly.nc"
        evaluation.to_file(
            filepath=input_file,
            model_name="WeeklyModel",
        )

        output_file = tmp_path / "metrics.csv"

        export_metrics(
            input_files=[input_file],
            output_file=output_file,
        )

        df = pd.read_csv(output_file)
        assert len(df) == 1
        assert df["model_name"].iloc[0] == "WeeklyModel"
