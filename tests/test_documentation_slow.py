"""Slow documentation tests - validates code examples from documentation.

Run with: pytest tests/test_documentation_slow.py -v --run-slow
"""

import importlib
import subprocess

import pytest

from tests.fixtures.doc_test_data import (
    CHAPKIT_REQUIRED_COLUMNS,
    CHAPKIT_RUN_INFO,
    CHAPKIT_TRAINING_DATA_MONTHLY,
    CHAPKIT_TRAINING_DATA_WEEKLY,
    CLI_HELP_COMMANDS,
    IMPORTABLE_CLASSES,
    SQLMODEL_EXAMPLES,
    TIME_PERIOD_PATTERNS,
    VALID_MODEL_CONFIG,
)


@pytest.mark.slow
class TestChapkitDataFormat:
    """Tests for chapkit.md JSON data format examples."""

    def test_weekly_training_data_has_required_columns(self):
        """Validate weekly training data has all required columns."""
        assert set(CHAPKIT_TRAINING_DATA_WEEKLY["columns"]) == set(CHAPKIT_REQUIRED_COLUMNS)

    def test_monthly_training_data_has_required_columns(self):
        """Validate monthly training data has all required columns."""
        assert set(CHAPKIT_TRAINING_DATA_MONTHLY["columns"]) == set(CHAPKIT_REQUIRED_COLUMNS)

    def test_weekly_time_period_format(self):
        """Validate weekly time period format (YYYY-Wnn)."""
        pattern = TIME_PERIOD_PATTERNS["weekly"]
        for row in CHAPKIT_TRAINING_DATA_WEEKLY["data"]:
            time_period = row[0]
            assert pattern.match(time_period), f"Invalid weekly format: {time_period}"

    def test_monthly_time_period_format(self):
        """Validate monthly time period format (YYYY-MM)."""
        pattern = TIME_PERIOD_PATTERNS["monthly"]
        for row in CHAPKIT_TRAINING_DATA_MONTHLY["data"]:
            time_period = row[0]
            assert pattern.match(time_period), f"Invalid monthly format: {time_period}"

    def test_run_info_structure(self):
        """Validate run_info JSON structure."""
        assert "prediction_length" in CHAPKIT_RUN_INFO
        assert isinstance(CHAPKIT_RUN_INFO["prediction_length"], int)
        assert "additional_continuous_covariates" in CHAPKIT_RUN_INFO
        assert isinstance(CHAPKIT_RUN_INFO["additional_continuous_covariates"], list)

    def test_model_config_structure(self):
        """Validate model configuration YAML structure."""
        assert "user_option_values" in VALID_MODEL_CONFIG
        assert isinstance(VALID_MODEL_CONFIG["user_option_values"], dict)


@pytest.mark.slow
class TestDocumentedImports:
    """Tests that documented imports actually work."""

    @pytest.mark.parametrize("module_path,name", IMPORTABLE_CLASSES)
    def test_import_documented_class(self, module_path, name):
        """Test that documented classes/functions can be imported."""
        module = importlib.import_module(module_path)
        assert hasattr(module, name), f"{name} not found in {module_path}"


@pytest.mark.slow
class TestCLIHelpCommands:
    """Tests that CLI --help commands work as documented."""

    @pytest.mark.parametrize("command", CLI_HELP_COMMANDS)
    def test_cli_help_command(self, command):
        """Test that CLI help commands execute successfully."""
        result = subprocess.run(command, capture_output=True, timeout=30)
        assert result.returncode == 0, f"Command {' '.join(command)} failed: {result.stderr.decode()}"
        assert len(result.stdout) > 0, f"No output from {' '.join(command)}"


@pytest.mark.slow
class TestSQLModelExamples:
    """Tests that SQLModel code examples from database_migrations.md compile."""

    @pytest.mark.parametrize("code", SQLMODEL_EXAMPLES)
    def test_sqlmodel_code_compiles(self, code):
        """Test that SQLModel code examples are valid Python syntax."""
        compile(code, "<string>", "exec")


@pytest.mark.slow
class TestDatasetExamples:
    """Tests for documented dataset operations."""

    def test_example_dataset_exists(self):
        """Test that example dataset referenced in docs exists."""
        from pathlib import Path

        example_csv = Path("example_data/laos_subset.csv")
        assert example_csv.exists(), f"Example dataset not found: {example_csv}"

    def test_example_geojson_exists(self):
        """Test that example GeoJSON referenced in docs exists."""
        from pathlib import Path

        example_geojson = Path("example_data/laos_subset.geojson")
        assert example_geojson.exists(), f"Example GeoJSON not found: {example_geojson}"

    def test_datasets_registry_contains_isimip(self):
        """Test that ISIMIP dataset referenced in docs is in registry."""
        from chap_core.file_io.example_data_set import datasets

        assert "ISIMIP_dengue_harmonized" in datasets
