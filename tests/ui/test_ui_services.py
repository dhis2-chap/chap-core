import pytest

from chap_core.api_types import BacktestParams
from chap_core.assessment.backtest_plots import list_backtest_plots
from chap_core.cli_endpoints.utils import compute_metrics_table
from chap_core.cli_endpoints.validate import collect_validation_issues
from chap_core.ui.services import (
    COMMAND_NAME,
    LOG_NAME,
    build_eval_command,
    format_cli_command,
    list_evaluations,
    make_plot,
    new_run_dir,
    start_eval,
)


def test_build_eval_command_uses_dotted_backtest_flags(tmp_path):
    args = build_eval_command(
        "my_model", "data.csv", tmp_path / "out.nc", BacktestParams(n_periods=2, n_splits=4, stride=3)
    )
    assert format_cli_command(args) == (
        f"chap eval --model-name my_model --dataset-csv data.csv --output-file {tmp_path / 'out.nc'} "
        "--backtest-params.n-periods 2 --backtest-params.n-splits 4 --backtest-params.stride 3"
    )


def test_build_eval_command_adds_model_configuration(tmp_path):
    args = build_eval_command("m", "d.csv", tmp_path / "o.nc", BacktestParams(), tmp_path / "config.yaml")
    assert args[-2:] == ["--model-configuration-yaml", str(tmp_path / "config.yaml")]


def test_new_run_dir_is_listed_once_it_has_an_evaluation(tmp_path):
    run_dir = new_run_dir(tmp_path, "https://github.com/dhis2-chap/minimalist_example_r/")
    assert run_dir.name.endswith("_minimalist-example-r")
    assert list_evaluations(tmp_path) == []
    (run_dir / "evaluation.nc").touch()
    assert list_evaluations(tmp_path) == [run_dir / "evaluation.nc"]


def test_start_eval_runs_chap_and_logs_output(tmp_path):
    proc = start_eval(["eval", "--help"], tmp_path)
    assert proc.wait(timeout=120) == 0
    assert (tmp_path / COMMAND_NAME).read_text() == "chap eval --help\n"
    assert "--model-name" in (tmp_path / LOG_NAME).read_text()


@pytest.mark.parametrize("plot_id", [plot["id"] for plot in list_backtest_plots()])
def test_make_plot_for_every_registered_plot(data_path, plot_id):
    chart = make_plot(data_path / "example_evaluation.nc", plot_id)
    assert chart.to_dict()


def test_compute_metrics_table_has_one_row_per_file(data_path):
    files = [data_path / "example_evaluation.nc", data_path / "example_evaluation_2.nc"]
    df = compute_metrics_table(files)
    assert list(df["filename"]) == ["example_evaluation.nc", "example_evaluation_2.nc"]


def test_collect_validation_issues_passes_for_example_dataset(data_path):
    issues = collect_validation_issues(str(data_path / "laos_subset.csv"))
    assert [issue for issue in issues if issue.level == "error"] == []


def test_collect_validation_issues_reports_missing_columns(data_path):
    issues = collect_validation_issues(str(data_path / "climate_data.csv"))
    assert any(issue.level == "error" and "Required column" in issue.message for issue in issues)
