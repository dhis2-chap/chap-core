import pytest

from chap_core.cli import app
from chap_core.cli_endpoints.generated_plot_ids import BACKTEST_PLOT_IDS
from chap_core.ui.catalog import covered_commands
from chap_core.ui.commands import build_args, command_fields, get_command, list_commands

PLACEHOLDERS = {"int": 1, "float": 1.0, "bool": True, "str": "x", "path": "x.csv", "list": ["x"]}


def test_every_cli_command_has_a_page_in_the_ui():
    assert set(list_commands()) == covered_commands()


@pytest.mark.parametrize("command", list_commands())
def test_filled_in_form_parses_as_cli_arguments(command):
    fields = command_fields(command)
    values = {f.key: f.default for f in fields}
    for field in fields:
        if field.required and values[field.key] in (None, ""):
            values[field.key] = field.choices[0] if field.choices else PLACEHOLDERS[field.kind]
    func, _, _ = app.parse_args(build_args(command, fields, values), exit_on_error=False)
    assert func is get_command(command).default_command


def test_build_args_leaves_out_cli_defaults():
    fields = command_fields("eval")
    values = {f.key: f.default for f in fields} | {"model_name": "m", "dataset_csv": "d.csv"}
    assert build_args("eval", fields, values) == [
        "eval",
        "--model-name",
        "m",
        "--dataset-csv",
        "d.csv",
        "--output-file",
        "evaluation.nc",
    ]


def test_build_args_uses_negative_flag_for_true_by_default_options():
    fields = command_fields("causal")
    values = {f.key: f.default for f in fields} | {"confidence_intervals": False}
    assert "--no-confidence-intervals" in build_args("causal", fields, values)


def test_build_args_sets_nested_options():
    fields = command_fields("eval")
    values = {"backtest_params.n_splits": 2, "run_config.debug": True, "estimator_options.mode": "hpo"}
    assert build_args("eval", fields, values) == [
        "eval",
        "--backtest-params.n-splits",
        "2",
        "--run-config.debug",
        "--estimator-options.mode",
        "hpo",
    ]


def test_build_args_repeats_list_options():
    fields = command_fields("export-metrics")
    args = build_args("export-metrics", fields, {"input_files": ["a.nc", "b.nc"], "output_file": "m.csv"})
    assert args == ["export-metrics", "--input-files", "a.nc", "--input-files", "b.nc", "--output-file", "m.csv"]


def test_ui_default_output_is_passed_even_when_optional_in_cli():
    fields = command_fields("model schema")
    values = {f.key: f.default for f in fields} | {"model_name": "m"}
    assert build_args("model schema", fields, values)[-2:] == ["--output-file", "model_schema.yaml"]


def test_plot_type_is_a_choice_of_registered_plots():
    plot_type = next(f for f in command_fields("plot-backtest") if f.key == "plot_type")
    assert plot_type.kind == "choice"
    assert plot_type.choices == tuple(BACKTEST_PLOT_IDS)
