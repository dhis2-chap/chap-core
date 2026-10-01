import pytest

from chap_core.assessment.backtest_plots import list_backtest_plots
from chap_core.cli_endpoints.utils import compute_metrics_table
from chap_core.cli_endpoints.validate import collect_validation_issues
from chap_core.plotting.dataset_plot import list_dataset_plots
from chap_core.ui.commands import command_fields
from chap_core.ui.services import (
    backtest_windows,
    command_name,
    format_cli_command,
    job_outputs,
    list_evaluations,
    list_jobs,
    make_dataset_plot,
    make_plot,
    option_value,
    plot_facets,
    resolve_paths,
    run_job,
    save_upload,
    workspace_files,
)


def test_format_cli_command_quotes_arguments():
    assert format_cli_command(["eval", "--model-name", "my model"]) == "chap eval --model-name 'my model'"


def test_successful_job_records_command_log_and_outputs(tmp_path, data_path):
    args = ["model", "schema", "--model-name", str(data_path.parent / "external_models/naive_python_model_uv")]
    job = run_job(tmp_path, [*args, "--output-file", "schema.yaml"], "model schema")
    assert job.status == "succeeded"
    assert job.exit_code == 0
    assert job.command.startswith("chap model schema")
    assert [p.name for p in job_outputs(job)] == ["schema.yaml"]
    assert list_jobs(tmp_path) == [job]


def test_failing_job_is_reported_as_failed(tmp_path, data_path):
    job = run_job(tmp_path, ["validate", "--dataset-csv", str(data_path / "climate_data.csv")], "validate")
    assert job.status == "failed"
    assert job.exit_code == 1
    assert "Required column" in job.log.read_text()


def test_resolve_paths_makes_inputs_absolute_and_keeps_outputs_relative(tmp_path):
    (tmp_path / "data.csv").touch()
    fields = command_fields("eval")
    values = {"dataset_csv": "data.csv", "output_file": "evaluation.nc", "model_name": "https://github.com/a/b"}
    resolved = resolve_paths(fields, values, tmp_path)
    assert resolved == values | {"dataset_csv": str(tmp_path / "data.csv")}


def test_workspace_files_lists_uploads_by_suffix(tmp_path):
    csv = save_upload(tmp_path, "data.csv", b"a,b\n")
    save_upload(tmp_path, "notes.txt", b"")
    assert csv in workspace_files(tmp_path, (".csv",))
    assert not [p for p in workspace_files(tmp_path, (".csv",)) if p.suffix == ".txt"]


def test_list_evaluations_is_empty_without_runs(tmp_path):
    assert list_evaluations(tmp_path) == []


@pytest.mark.parametrize("plot_id", [plot["id"] for plot in list_backtest_plots()])
def test_make_plot_for_every_registered_plot(data_path, plot_id):
    chart = make_plot(data_path / "example_evaluation.nc", plot_id)
    assert chart.to_dict()


@pytest.mark.parametrize("plot_id", [plot["id"] for plot in list_dataset_plots()])
def test_make_dataset_plot_for_every_registered_plot(data_path, plot_id):
    chart = make_dataset_plot(data_path / "laos_subset.csv", plot_id)
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


def test_backtest_windows_count_back_from_the_end_of_the_data():
    periods = [f"2020-{m:02d}" for m in range(1, 13)]
    windows = backtest_windows(periods, n_periods=3, n_splits=2, stride=1)
    assert windows == [
        {"split": 1, "train_end": "2020-08", "forecast_start": "2020-09", "forecast_end": "2020-11"},
        {"split": 2, "train_end": "2020-09", "forecast_start": "2020-10", "forecast_end": "2020-12"},
    ]


def test_backtest_windows_is_empty_when_the_data_is_too_short():
    assert backtest_windows(["2020-01", "2020-02"], n_periods=3, n_splits=1, stride=1) == []


def test_faceted_plot_can_show_a_single_cell(data_path):
    nc = data_path / "example_evaluation.nc"
    facets = plot_facets(nc, "evaluation_plot")
    assert [column for column, _, _ in facets] == ["split_period", "location"]
    chart = make_plot(nc, "evaluation_plot", {column: values[0] for column, _, values in facets})
    assert chart.to_dict()


def test_command_name_and_option_value_read_job_arguments():
    args = ["causal", "build-counterfactual", "--dataset-csv", "d.csv"]
    assert command_name(args) == "causal build-counterfactual"
    assert option_value(args, "--dataset-csv") == "d.csv"
    assert option_value(args, "--model-name") is None


@pytest.mark.parametrize(("n_periods", "n_splits", "stride"), [(3, 7, -1), (3, 7, 0), (0, 7, 1), (3, 0, 1)])
def test_backtest_windows_is_empty_for_values_chap_eval_rejects(n_periods, n_splits, stride):
    periods = [f"2020-{m:02d}" for m in range(1, 13)]
    assert backtest_windows(periods, n_periods, n_splits, stride) == []
