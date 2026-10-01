"""Streamlit-free helpers behind the chap UI pages: the runs folder, uploads and background jobs."""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import re
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chap_core.ui.commands import Field

LOG_NAME = "output.log"
COMMAND_NAME = "command.txt"
ARGS_NAME = "args.json"
PID_NAME = "pid"
EXIT_CODE_NAME = "exit_code"
STOPPED_NAME = "stopped"
JOB_FILES = {LOG_NAME, COMMAND_NAME, ARGS_NAME, PID_NAME, EXIT_CODE_NAME, STOPPED_NAME}

# Only present in a source checkout, not in an installed wheel
REPO_ROOT = Path(__file__).parent.parent.parent
EXAMPLE_DATA_DIR = REPO_ROOT / "example_data"
EXAMPLE_MODELS_DIR = REPO_ROOT / "external_models"
EXAMPLE_DATASETS = ["laos_subset.csv", "Minimalist_multiregion_example_data.csv", "nicaragua_weekly_subset.csv"]
EXAMPLE_EVALUATIONS = ["example_evaluation.nc", "example_evaluation_2.nc"]
EXAMPLE_MODEL = EXAMPLE_MODELS_DIR / "naive_python_model_uv"

# Datasets published with region polygons, offered as examples in any installation.
PUBLISHED_DATASETS = {
    "Laos, provinces, monthly": "https://raw.githubusercontent.com/dhis2/climate-health-data/main/lao/chap_LAO_admin1_monthly.csv",
    "Thailand, provinces, monthly": "https://raw.githubusercontent.com/dhis2/climate-health-data/main/tha/chap_THA_admin1_monthly.csv",
    "Vietnam, provinces, monthly": "https://raw.githubusercontent.com/dhis2/climate-health-data/main/vnm/chap_VNM_admin1_monthly.csv",
}

# Popen handles of jobs started by this server process, so finished children are reaped.
_processes: dict[Path, subprocess.Popen] = {}


def get_runs_dir() -> Path:
    """The runs folder, shared with the chap CLI: CHAP_RUNS_DIR, else ./runs.

    It holds one folder per run, plus uploads, saved configurations and the models' own working
    folders. It is created when something is first written to it, not when the UI starts.
    """
    return Path(os.environ.get("CHAP_RUNS_DIR", "runs")).resolve()


def get_uploads_dir() -> Path:
    """Where files added through the browser are kept: CHAP_UPLOADS_DIR, else <runs folder>/uploads."""
    uploads = os.environ.get("CHAP_UPLOADS_DIR")
    return Path(uploads).resolve() if uploads else get_runs_dir() / "uploads"


def example_files(names: list[str]) -> list[Path]:
    """The named files from the repository's example_data directory that exist."""
    return [EXAMPLE_DATA_DIR / name for name in names if (EXAMPLE_DATA_DIR / name).exists()]


def example_models() -> list[Path]:
    """Model directories bundled with the repository that have an MLproject file."""
    if not EXAMPLE_MODELS_DIR.exists():
        return []
    return sorted(p.parent for p in EXAMPLE_MODELS_DIR.glob("*/MLproject"))


def save_upload(uploads_dir: Path, name: str, content: bytes) -> Path:
    """Store a file added through the browser in the uploads folder and return its path."""
    path = uploads_dir / Path(name).name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def fetch_published_dataset(uploads_dir: Path, url: str) -> Path:
    """Download a published dataset and its polygons into the uploads folder, once, and return the CSV."""
    import shutil

    from chap_core.cli_endpoints._common import resolve_csv_path

    target = uploads_dir / url.rsplit("/", 1)[-1]
    if not target.exists():
        csv_path, geojson_path = resolve_csv_path(url)
        uploads_dir.mkdir(parents=True, exist_ok=True)
        if geojson_path is not None:
            shutil.copyfile(geojson_path, target.with_suffix(".geojson"))
        shutil.copyfile(csv_path, target)
    return target


def workspace_files(runs_dir: Path, uploads_dir: Path, suffixes: tuple[str, ...]) -> list[Path]:
    """Uploaded files, saved configs, run outputs and example files with one of the given suffixes."""
    folders = (uploads_dir, runs_dir / "configs")
    files = sorted(p for folder in folders for p in folder.glob("*") if p.suffix in suffixes)
    for job in list_jobs(runs_dir):
        files += [p for p in job_outputs(job) if p.suffix in suffixes]
    if EXAMPLE_DATA_DIR.exists():
        files += sorted(p for p in EXAMPLE_DATA_DIR.glob("*") if p.suffix in suffixes)
    return files


def format_cli_command(args: list[str]) -> str:
    """The `chap ...` command a user could paste into a terminal."""
    return shlex.join(["chap", *args])


def resolve_paths(fields: list[Field], values: dict[str, Any], cwd: Path) -> dict[str, Any]:
    """Make relative input paths absolute, since jobs run inside their own run directory.

    Output paths stay relative so the files land in the run directory.
    """

    def resolve(value):
        if isinstance(value, str | Path) and str(value) and not Path(value).is_absolute():
            candidate = cwd / value
            if candidate.exists():
                return str(candidate.resolve())
        return value

    resolved = dict(values)
    for field in fields:
        value = values.get(field.key)
        if field.is_output or value is None or field.kind in ("bool", "int", "float", "choice"):
            continue
        resolved[field.key] = [resolve(v) for v in value] if isinstance(value, list) else resolve(value)
    return resolved


@dataclass(frozen=True)
class Job:
    run_dir: Path
    args: list[str]
    status: str  # running | succeeded | failed | stopped
    exit_code: int | None
    started: datetime
    finished: datetime | None

    @property
    def name(self) -> str:
        return self.run_dir.name

    @property
    def command(self) -> str:
        return format_cli_command(self.args)

    @property
    def log(self) -> Path:
        return self.run_dir / LOG_NAME


def start_job(runs_dir: Path, args: list[str], label: str) -> Job:
    """Run a chap command in the background in a fresh folder of the runs folder."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", label).strip("-").lower() or "job"
    stamp = f"{datetime.now():%Y%m%d-%H%M%S-%f}"[:-3]
    run_dir = runs_dir / f"{stamp}_{slug}"
    run_dir.mkdir(parents=True)
    (run_dir / ARGS_NAME).write_text(json.dumps(args))
    (run_dir / COMMAND_NAME).write_text(format_cli_command(args) + "\n")
    env = _child_env(runs_dir)
    # Start the job with posix_spawn rather than fork: forking the UI process, which has threads and
    # libraries such as PROJ loaded, can crash the child in their fork handlers before it starts.
    # Python only uses posix_spawn without cwd= or start_new_session=, so the runner changes into
    # its run directory and starts its own session itself. File descriptors Python opens are not
    # inheritable, so not closing them only passes on the log and stdin set here.
    with open(run_dir / LOG_NAME, "w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "chap_core.ui.job_runner", str(run_dir), *args],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            close_fds=False,
        )
    (run_dir / PID_NAME).write_text(str(proc.pid))
    _processes[run_dir] = proc
    return load_job(run_dir)


def _child_env(runs_dir: Path) -> dict[str, str]:
    """Environment for processes the UI starts; models keep their working folders in the runs folder."""
    return {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg",
        "CHAP_RUNS_DIR": str(runs_dir),
    }


def run_external(argv: list[str], timeout: float = 1800) -> subprocess.CompletedProcess:
    """Run another program, such as chaps, from the UI and wait for it.

    Goes through chap_core.ui.spawn so it is started with posix_spawn and does not keep the UI's
    socket (see start_job).
    """
    return subprocess.run(
        [sys.executable, "-m", "chap_core.ui.spawn", *argv],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        close_fds=False,
        timeout=timeout,
    )


def validate_against_model(runs_dir: Path, dataset_csv: str, model_name: str, timeout: float = 600) -> list[dict]:
    """Validation issues for a dataset and a model, found in a separate process.

    Loading the model can start other programs, which must not be forked from the UI process
    (see start_job), so this runs `chap_core.ui.validation_runner` the same way jobs are started.
    """
    result = subprocess.run(
        [sys.executable, "-m", "chap_core.ui.validation_runner", dataset_csv, model_name],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env=_child_env(runs_dir),
        close_fds=False,
        timeout=timeout,
    )
    if result.returncode != 0:
        lines = (result.stderr or result.stdout).strip().splitlines()
        raise RuntimeError(lines[-1] if lines else f"validation exited with code {result.returncode}")
    issues: list[dict] = json.loads(result.stdout.strip().splitlines()[-1])
    return issues


def run_job(runs_dir: Path, args: list[str], label: str, timeout: float = 600) -> Job:
    """Run a chap command like `start_job`, but wait for it to finish."""
    job = start_job(runs_dir, args, label)
    _processes[job.run_dir].wait(timeout=timeout)
    return load_job(job.run_dir)


def load_job(run_dir: Path) -> Job:
    """Current state of a job from the files in its run directory."""
    args = json.loads((run_dir / ARGS_NAME).read_text())
    started = datetime.fromtimestamp((run_dir / ARGS_NAME).stat().st_mtime)
    exit_file = run_dir / EXIT_CODE_NAME
    proc = _processes.get(run_dir)
    if proc is not None:
        proc.poll()
    if exit_file.exists():
        exit_code = int(exit_file.read_text() or 1)
        status = "succeeded" if exit_code == 0 else "failed"
        return Job(run_dir, args, status, exit_code, started, datetime.fromtimestamp(exit_file.stat().st_mtime))
    if (run_dir / STOPPED_NAME).exists():
        finished = datetime.fromtimestamp((run_dir / STOPPED_NAME).stat().st_mtime)
        return Job(run_dir, args, "stopped", None, started, finished)
    if _is_running(run_dir, proc):
        return Job(run_dir, args, "running", None, started, None)
    return Job(run_dir, args, "failed", None, started, None)


def list_jobs(runs_dir: Path) -> list[Job]:
    """All runs started from the UI, newest first. The models' own working folders are not runs."""
    run_dirs = sorted((p.parent for p in runs_dir.glob(f"*/{ARGS_NAME}")), reverse=True)
    return [load_job(run_dir) for run_dir in run_dirs]


def stop_job(job: Job) -> None:
    """Stop a running job and everything it started."""
    (job.run_dir / STOPPED_NAME).touch()
    with contextlib.suppress(ProcessLookupError, PermissionError, ValueError):
        os.killpg(int((job.run_dir / PID_NAME).read_text()), signal.SIGTERM)


def job_outputs(job: Job) -> list[Path]:
    """Files a job wrote into its run directory."""
    return sorted(
        p for p in job.run_dir.rglob("*") if p.is_file() and p.name not in JOB_FILES and not p.name.startswith(".")
    )


def list_evaluations(runs_dir: Path) -> list[Path]:
    """Evaluation files written by successful jobs, newest first."""
    return [
        p for job in list_jobs(runs_dir) if job.status == "succeeded" for p in job_outputs(job) if p.suffix == ".nc"
    ]


def make_plot(nc_path: Path, plot_id: str, coords: dict[str, Any] | None = None):
    """Altair chart for one registered backtest plot of an evaluation file.

    With `coords` (e.g. one split period and one location) a faceted plot shows only that cell.
    """
    from chap_core.assessment.backtest_plots import FacetedBacktestPlot, create_plot_from_evaluation, get_backtest_plot
    from chap_core.assessment.evaluation import Evaluation

    evaluation = Evaluation.from_file(nc_path)
    plot_cls = get_backtest_plot(plot_id)
    if not coords or plot_cls is None or not issubclass(plot_cls, FacetedBacktestPlot):
        return create_plot_from_evaluation(plot_id, evaluation)
    flat = evaluation.to_flat()
    historical = flat.historical_observations if plot_cls.needs_historical else None
    return plot_cls().get_subplot(flat.observations, flat.forecasts, coords, historical)


def plot_facets(nc_path: Path, plot_id: str) -> list[tuple[str, str, list[Any]]]:
    """The dimensions a plot is faceted by, as (column, display name, values); empty for unfaceted plots."""
    from chap_core.assessment.backtest_plots import FacetedBacktestPlot, get_backtest_plot
    from chap_core.assessment.evaluation import Evaluation

    plot_cls = get_backtest_plot(plot_id)
    if plot_cls is None or not issubclass(plot_cls, FacetedBacktestPlot):
        return []
    flat = Evaluation.from_file(nc_path).to_flat()
    historical = flat.historical_observations if plot_cls.needs_historical else None
    plot = plot_cls()
    coords = plot.facet_coords(flat.observations, flat.forecasts, historical)
    return [
        (dim.clean_name, dim.display_name, coords[dim.clean_name])
        for dim in plot.facet_dimensions
        if dim.clean_name in coords
    ]


def backtest_windows(periods: list[str], n_periods: int, n_splits: int, stride: int) -> list[dict[str, Any]]:
    """Training and forecast window of each backtest split, as `chap eval` cuts them.

    Mirrors `train_test_generator`: splits are counted back from the end of the data, and split i
    trains on everything up to `first_train_end + i * stride` and forecasts the next `n_periods`.
    """
    if min(n_periods, n_splits, stride) < 1:
        return []
    first_train_end = len(periods) - n_periods - (n_splits - 1) * stride - 1
    if first_train_end < 0:
        return []
    windows = []
    for i in range(n_splits):
        train_end = first_train_end + i * stride
        windows.append(
            {
                "split": i + 1,
                "train_end": periods[train_end],
                "forecast_start": periods[train_end + 1],
                "forecast_end": periods[train_end + n_periods],
            }
        )
    return windows


def option_value(args: list[str], flag: str) -> str | None:
    """Value given for a CLI option in a job's arguments."""
    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else None


def command_name(args: list[str]) -> str:
    """The chap command a job ran, e.g. `eval` or `causal build-counterfactual`."""
    return " ".join(itertools.takewhile(lambda a: not a.startswith("-"), args))


def dataset_geojson(csv_path: Path) -> dict | None:
    """Region polygons stored next to a dataset as `<name>.geojson`, if there are any."""
    path = Path(csv_path).with_suffix(".geojson")
    return json.loads(path.read_text()) if path.exists() else None


def dataset_incidence(csv_path: Path) -> tuple[dict[str, float], str]:
    """Value per location for a dataset map: annual incidence per 1000 when population is known, else mean cases.

    Uses the same numbers as the "Disease Cases Map" dataset plot.
    """
    from chap_core.plotting.dataset_plot import get_dataset_plot
    from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

    plot_cls = get_dataset_plot("disease-cases-map")
    assert plot_cls is not None
    plot = plot_cls.from_dataset(DataSet.from_csv(csv_path))
    data = plot.data()
    column = data.columns[-1]
    label = "Annual incidence per 1000" if column == "annual_incidence_per_1000" else "Mean disease cases"
    return dict(zip(data["location"].astype(str), data[column].astype(float), strict=True)), label


def metric_by_location(nc_path: Path, metric_id: str) -> dict[str, float]:
    """One metric of an evaluation, aggregated per location."""
    from chap_core.assessment.evaluation import Evaluation
    from chap_core.assessment.flat_representations import DataDimension
    from chap_core.assessment.metrics import get_metric

    metric_cls = get_metric(metric_id)
    if metric_cls is None:
        raise ValueError(f"Unknown metric: {metric_id}")
    flat = Evaluation.from_file(nc_path).to_flat()
    values = metric_cls().get_metric(flat.observations, flat.forecasts, dimensions=(DataDimension.location,))
    return dict(zip(values["location"].astype(str), values["metric"].astype(float), strict=True))


def evaluation_dataset(nc_path: Path) -> str | None:
    """The dataset an evaluation was made from, when it was made by a UI run."""
    run_dir = Path(nc_path).parent
    if not (run_dir / ARGS_NAME).exists():
        return None
    return option_value(load_job(run_dir).args, "--dataset-csv")


def make_dataset_plot(csv_path: Path, plot_id: str):
    """Altair chart for one registered dataset plot of a CSV dataset."""
    from chap_core.plotting.dataset_plot import get_dataset_plot
    from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

    plot_cls = get_dataset_plot(plot_id)
    if plot_cls is None:
        raise ValueError(f"Unknown dataset plot: {plot_id}")
    return plot_cls.from_dataset(DataSet.from_csv(csv_path)).plot()


def _is_running(run_dir: Path, proc: subprocess.Popen | None) -> bool:
    if proc is not None:
        return proc.returncode is None
    try:
        os.kill(int((run_dir / PID_NAME).read_text()), 0)
    except (ProcessLookupError, PermissionError, ValueError, FileNotFoundError):
        return False
    return True
