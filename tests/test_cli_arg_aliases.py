"""Tests for the shared CLI argument annotations in chap_core.cli_endpoints._args.

These tests lock in that commands using the shared aliases (e.g. ModelNameArg,
DatasetCsvArg) end up with the canonical help text in their resolved type hints,
so help strings stay in sync across the CLI.
"""

from __future__ import annotations

from typing import Annotated, get_args, get_type_hints

import pytest

from chap_core.cli_endpoints._args import (
    BacktestParamsArg,
    DatasetCsvArg,
    DataSourceMappingArg,
    ModelConfigYamlArg,
    ModelNameArg,
    RunConfigArg,
)
from chap_core.cli_endpoints.causal import causal_cmd
from chap_core.cli_endpoints.evaluate import eval_cmd
from chap_core.cli_endpoints.explain import explain_lime
from chap_core.cli_endpoints.report import report
from chap_core.cli_endpoints.validate import validate_cmd


def _help_text(annotation) -> str:
    """Extract the cyclopts Parameter help string from an Annotated type."""
    metadata = get_args(annotation)[1:]
    for item in metadata:
        if hasattr(item, "help") and item.help:
            return item.help
    raise AssertionError(f"No Parameter help string found on {annotation!r}")


@pytest.mark.parametrize(
    ("alias", "expected_substring"),
    [
        (ModelNameArg, "Model path (local directory), GitHub URL, or chapkit service URL"),
        (DatasetCsvArg, "Path or URL to CSV file"),
        (RunConfigArg, "Model execution configuration"),
        (BacktestParamsArg, "Backtest configuration"),
        (ModelConfigYamlArg, "YAML file with model-specific configuration parameters"),
        (DataSourceMappingArg, "JSON file mapping model covariate names to CSV column names"),
    ],
)
def test_alias_help_strings(alias, expected_substring):
    assert expected_substring in _help_text(alias)


@pytest.mark.parametrize(
    ("command", "param_name", "expected_substring"),
    [
        (eval_cmd, "model_name", "GitHub URL, or chapkit service URL"),
        (eval_cmd, "dataset_csv", "Path or URL to CSV file"),
        (eval_cmd, "run_config", "Model execution configuration"),
        (eval_cmd, "backtest_params", "Backtest configuration"),
        (causal_cmd, "model_name", "GitHub URL, or chapkit service URL"),
        (causal_cmd, "dataset_csv", "Path or URL to CSV file"),
        (validate_cmd, "dataset_csv", "Path or URL to CSV file"),
        (explain_lime, "model_name", "GitHub URL, or chapkit service URL"),
        (explain_lime, "dataset_csv", "Path or URL to CSV file"),
        (report, "model_name", "GitHub URL, or chapkit service URL"),
        (report, "dataset_csv", "Path or URL to CSV file"),
    ],
)
def test_commands_inherit_shared_help(command, param_name, expected_substring):
    hints = get_type_hints(command, include_extras=True)
    annotation = hints[param_name]
    assert expected_substring in _help_text(annotation)


def test_renamed_outliers_no_longer_use_model_path():
    """report and explain previously exposed `model_path`; they now use `model_name`."""
    for command in (report, explain_lime):
        hints = get_type_hints(command, include_extras=True)
        assert "model_name" in hints
        assert "model_path" not in hints


def test_aliases_are_annotated_types():
    """Sanity-check that each alias is an Annotated[...] with cyclopts Parameter metadata."""
    for alias in (
        ModelNameArg,
        DatasetCsvArg,
        RunConfigArg,
        BacktestParamsArg,
        ModelConfigYamlArg,
        DataSourceMappingArg,
    ):
        # Annotated unwraps to (origin_type, *metadata)
        args = get_args(alias)
        assert len(args) >= 2, f"{alias!r} is not a properly formed Annotated type"
