"""Integration tests for CLI workflow documented in evaluation-workflow.md."""

import subprocess
import tempfile
from pathlib import Path

import pytest

EXAMPLE_DATA = Path("example_data/laos_subset.csv")
# Use a lightweight GitHub model for testing (evaluate2 requires path or URL, not built-in names)
TEST_MODEL = "https://github.com/dhis2-chap/minimalist_example_lag"


@pytest.mark.slow
def test_evaluate2_with_github_model():
    """Test evaluate2 command with GitHub model and example data."""
    if not EXAMPLE_DATA.exists():
        pytest.skip(f"Example data not found: {EXAMPLE_DATA}")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_nc = Path(tmpdir) / "eval.nc"

        result = subprocess.run(
            [
                "chap",
                "evaluate2",
                "--model-name",
                TEST_MODEL,
                "--dataset-csv",
                str(EXAMPLE_DATA),
                "--output-file",
                str(output_nc),
                "--backtest-params.n-splits",
                "2",
                "--backtest-params.n-periods",
                "1",
            ],
            capture_output=True,
            timeout=300,
        )
        assert result.returncode == 0, f"evaluate2 failed: {result.stderr.decode()}"
        assert output_nc.exists(), "Output NetCDF file was not created"


@pytest.mark.slow
def test_plot_backtest():
    """Test plot-backtest command with evaluate2 output."""
    if not EXAMPLE_DATA.exists():
        pytest.skip(f"Example data not found: {EXAMPLE_DATA}")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_nc = Path(tmpdir) / "eval.nc"
        output_html = Path(tmpdir) / "plot.html"

        # First run evaluate2
        result = subprocess.run(
            [
                "chap",
                "evaluate2",
                "--model-name",
                TEST_MODEL,
                "--dataset-csv",
                str(EXAMPLE_DATA),
                "--output-file",
                str(output_nc),
                "--backtest-params.n-splits",
                "2",
                "--backtest-params.n-periods",
                "1",
            ],
            capture_output=True,
            timeout=300,
        )
        assert result.returncode == 0, f"evaluate2 failed: {result.stderr.decode()}"

        # Then run plot-backtest
        result = subprocess.run(
            [
                "chap",
                "plot-backtest",
                "--input-file",
                str(output_nc),
                "--output-file",
                str(output_html),
            ],
            capture_output=True,
            timeout=60,
        )
        assert result.returncode == 0, f"plot-backtest failed: {result.stderr.decode()}"
        assert output_html.exists(), "Output HTML file was not created"


@pytest.mark.slow
def test_export_metrics():
    """Test export-metrics command with evaluate2 output."""
    if not EXAMPLE_DATA.exists():
        pytest.skip(f"Example data not found: {EXAMPLE_DATA}")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_nc = Path(tmpdir) / "eval.nc"
        output_csv = Path(tmpdir) / "metrics.csv"

        # First run evaluate2
        result = subprocess.run(
            [
                "chap",
                "evaluate2",
                "--model-name",
                TEST_MODEL,
                "--dataset-csv",
                str(EXAMPLE_DATA),
                "--output-file",
                str(output_nc),
                "--backtest-params.n-splits",
                "2",
                "--backtest-params.n-periods",
                "1",
            ],
            capture_output=True,
            timeout=300,
        )
        assert result.returncode == 0, f"evaluate2 failed: {result.stderr.decode()}"

        # Then run export-metrics
        result = subprocess.run(
            [
                "chap",
                "export-metrics",
                "--input-files",
                str(output_nc),
                "--output-file",
                str(output_csv),
            ],
            capture_output=True,
            timeout=60,
        )
        assert result.returncode == 0, f"export-metrics failed: {result.stderr.decode()}"
        assert output_csv.exists(), "Output CSV file was not created"

        # Verify CSV has content
        content = output_csv.read_text()
        assert "minimalist_example_lag" in content or "filename" in content


@pytest.mark.slow
def test_full_evaluation_workflow():
    """Test the complete evaluation workflow from docs."""
    if not EXAMPLE_DATA.exists():
        pytest.skip(f"Example data not found: {EXAMPLE_DATA}")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_nc = Path(tmpdir) / "eval.nc"
        output_html = Path(tmpdir) / "plot.html"
        output_csv = Path(tmpdir) / "metrics.csv"

        # Step 1: evaluate2
        result = subprocess.run(
            [
                "chap",
                "evaluate2",
                "--model-name",
                TEST_MODEL,
                "--dataset-csv",
                str(EXAMPLE_DATA),
                "--output-file",
                str(output_nc),
                "--backtest-params.n-splits",
                "2",
                "--backtest-params.n-periods",
                "1",
            ],
            capture_output=True,
            timeout=300,
        )
        assert result.returncode == 0, f"evaluate2 failed: {result.stderr.decode()}"
        assert output_nc.exists()

        # Step 2: plot-backtest
        result = subprocess.run(
            [
                "chap",
                "plot-backtest",
                "--input-file",
                str(output_nc),
                "--output-file",
                str(output_html),
            ],
            capture_output=True,
            timeout=60,
        )
        assert result.returncode == 0, f"plot-backtest failed: {result.stderr.decode()}"
        assert output_html.exists()

        # Step 3: export-metrics
        result = subprocess.run(
            [
                "chap",
                "export-metrics",
                "--input-files",
                str(output_nc),
                "--output-file",
                str(output_csv),
            ],
            capture_output=True,
            timeout=60,
        )
        assert result.returncode == 0, f"export-metrics failed: {result.stderr.decode()}"
        assert output_csv.exists()
