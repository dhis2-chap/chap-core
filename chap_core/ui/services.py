"""Streamlit-free helpers behind the chap UI pages: workspace files and background jobs."""

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

# Popen handles of jobs started by this server process, so finished children are reaped.
_processes: dict[Path, subprocess.Popen] = {}


def get_workdir() -> Path:
    """Workdir the UI was started with (`chap ui --workdir`), falling back to ./chap-ui."""
    workdir = Path(os.environ.get("CHAP_UI_WORKDIR", "chap-ui")).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def example_files(names: list[str]) -> list[Path]:
    """The named files from the repository's example_data directory that exist."""
    return [EXAMPLE_DATA_DIR / name for name in names if (EXAMPLE_DATA_DIR / name).exists()]


def example_models() -> list[Path]:
    """Model directories bundled with the repository that have an MLproject file."""
    if not EXAMPLE_MODELS_DIR.exists():
        return []
    return sorted(p.parent for p in EXAMPLE_MODELS_DIR.glob("*/MLproject"))


def save_upload(workdir: Path, name: str, content: bytes) -> Path:
    """Store an uploaded file under `<workdir>/uploads/` and return its path."""
    path = workdir / "uploads" / Path(name).name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def workspace_files(workdir: Path, suffixes: tuple[str, ...]) -> list[Path]:
    """Uploaded files, saved configs, run outputs and example files with one of the given suffixes."""
    files = sorted(p for folder in ("uploads", "configs") for p in (workdir / folder).glob("*") if p.suffix in suffixes)
    for job in list_jobs(workdir):
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


def start_job(workdir: Path, args: list[str], label: str) -> Job:
    """Run a chap command in the background in a fresh run directory under `<workdir>/runs/`."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", label).strip("-").lower() or "job"
    stamp = f"{datetime.now():%Y%m%d-%H%M%S-%f}"[:-3]
    run_dir = workdir / "runs" / f"{stamp}_{slug}"
    run_dir.mkdir(parents=True)
    (run_dir / ARGS_NAME).write_text(json.dumps(args))
    (run_dir / COMMAND_NAME).write_text(format_cli_command(args) + "\n")
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg",
        "CHAP_RUNS_DIR": str(workdir / "model-runs"),
    }
    with open(run_dir / LOG_NAME, "w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "chap_core.ui.job_runner", str(run_dir), *args],
            cwd=run_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
    (run_dir / PID_NAME).write_text(str(proc.pid))
    _processes[run_dir] = proc
    return load_job(run_dir)


def run_job(workdir: Path, args: list[str], label: str, timeout: float = 600) -> Job:
    """Run a chap command like `start_job`, but wait for it to finish."""
    job = start_job(workdir, args, label)
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


def list_jobs(workdir: Path) -> list[Job]:
    """All jobs in the workdir, newest first."""
    run_dirs = sorted((p.parent for p in (workdir / "runs").glob(f"*/{ARGS_NAME}")), reverse=True)
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


def list_evaluations(workdir: Path) -> list[Path]:
    """Evaluation files written by successful jobs, newest first."""
    return [p for job in list_jobs(workdir) if job.status == "succeeded" for p in job_outputs(job) if p.suffix == ".nc"]


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
