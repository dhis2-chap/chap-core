import pytest

from chap_core.assessment.backtest_plots import list_backtest_plots
from chap_core.cli_endpoints.utils import compute_metrics_table
from chap_core.cli_endpoints.validate import collect_validation_issues
from chap_core.plotting.dataset_plot import list_dataset_plots
from chap_core.ui.commands import command_fields
from chap_core.ui.services import (
    format_cli_command,
    job_outputs,
    list_evaluations,
    list_jobs,
    make_dataset_plot,
    make_plot,
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
