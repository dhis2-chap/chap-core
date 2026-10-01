import fcntl
import os
import subprocess
import sys
import time

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
    get_runs_dir,
    get_uploads_dir,
    job_outputs,
    list_evaluations,
    list_jobs,
    load_job,
    make_dataset_plot,
    make_plot,
    option_value,
    plot_facets,
    resolve_paths,
    run_job,
    save_upload,
    stop_job,
    validate_against_model,
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
    uploads = tmp_path / "uploads"
    csv = save_upload(uploads, "data.csv", b"a,b\n")
    save_upload(uploads, "notes.txt", b"")
    assert csv == uploads / "data.csv"
    files = workspace_files(tmp_path, uploads, (".csv",))
    assert csv in files
    assert not [p for p in files if p.suffix == ".txt"]


def test_runs_folder_is_shared_with_the_cli_and_not_created_by_reading_it(monkeypatch, tmp_path):
    monkeypatch.delenv("CHAP_RUNS_DIR", raising=False)
    monkeypatch.delenv("CHAP_UPLOADS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert get_runs_dir() == tmp_path / "runs"
    assert get_uploads_dir() == tmp_path / "runs" / "uploads"
    assert list_jobs(get_runs_dir()) == []
    assert not (tmp_path / "runs").exists()
    monkeypatch.setenv("CHAP_UPLOADS_DIR", str(tmp_path / "data"))
    assert get_uploads_dir() == tmp_path / "data"


def test_jobs_get_their_own_folder_in_the_runs_folder(tmp_path, data_path):
    job = run_job(tmp_path, ["validate", "--dataset-csv", str(data_path / "laos_subset.csv")], "validate")
    assert job.run_dir.parent == tmp_path


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


def test_validate_against_model_runs_in_a_separate_process(tmp_path, data_path, models_path):
    issues = validate_against_model(
        tmp_path, str(data_path / "laos_subset.csv"), str(models_path / "naive_python_model_uv")
    )
    assert issues
    assert all(issue["level"] == "warning" for issue in issues)


def test_validate_against_model_explains_an_unreachable_service(tmp_path, data_path):
    with pytest.raises(RuntimeError, match="could not be reached as a chapkit service"):
        validate_against_model(tmp_path, str(data_path / "laos_subset.csv"), "http://localhost:1")


def test_helpers_close_descriptors_inherited_from_the_ui():
    read_end, write_end = os.pipe()
    os.set_inheritable(read_end, True)
    try:
        code = (
            "import os; from chap_core.ui.job_runner import close_inherited_descriptors; "
            f"close_inherited_descriptors(); os.fstat({read_end})"
        )
        result = subprocess.run([sys.executable, "-c", code], close_fds=False, capture_output=True, text=True)
    finally:
        os.close(read_end)
        os.close(write_end)
    assert result.returncode != 0
    assert "Bad file descriptor" in result.stderr


def _dead_run_dir(runs_dir, pid: int):
    """A run whose runner died without writing an exit code, its process id now belonging to `pid`."""
    run_dir = runs_dir / "20260101-000000-000_eval"
    run_dir.mkdir(parents=True)
    (run_dir / "args.json").write_text('["eval"]')
    (run_dir / "pid").write_text(str(pid))
    (run_dir / "running.lock").touch()
    old = time.time() - 3600
    os.utime(run_dir / "args.json", (old, old))
    return run_dir


def test_a_reused_process_id_does_not_keep_a_dead_job_running_or_get_stopped(tmp_path):
    unrelated = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        run_dir = _dead_run_dir(tmp_path, unrelated.pid)
        job = load_job(run_dir)
        assert job.status == "failed"
        stop_job(job)
        assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait()


def test_a_job_counts_as_running_while_its_runner_holds_the_lock(tmp_path):
    run_dir = _dead_run_dir(tmp_path, os.getpid())
    with open(run_dir / "running.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert load_job(run_dir).status == "running"
    assert load_job(run_dir).status == "failed"


def test_stop_reaches_a_runner_that_has_not_started_its_own_session_yet(tmp_path):
    runner = subprocess.Popen(["sleep", "30"])  # not a process group leader, like a runner before setsid
    try:
        run_dir = tmp_path / "20260101-000000-000_eval"
        run_dir.mkdir()
        (run_dir / "args.json").write_text('["eval"]')
        (run_dir / "pid").write_text(str(runner.pid))
        job = load_job(run_dir)
        assert job.status == "running"
        stop_job(job)
        assert runner.wait(timeout=10) == -15
    finally:
        runner.kill()
        runner.wait()


def test_run_job_stops_a_job_that_takes_too_long(tmp_path, data_path):
    job = run_job(tmp_path, ["validate", "--dataset-csv", str(data_path / "laos_subset.csv")], "validate", timeout=0.01)
    assert job.status == "stopped"
