"""Streamlit-free helpers behind the chap UI pages."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chap_core.api_types import BacktestParams

EVAL_OUTPUT_NAME = "evaluation.nc"
LOG_NAME = "eval.log"
COMMAND_NAME = "command.txt"

# Only present in a source checkout, not in an installed wheel
EXAMPLE_DATA_DIR = Path(__file__).parent.parent.parent / "example_data"
EXAMPLE_DATASETS = ["laos_subset.csv", "Minimalist_multiregion_example_data.csv", "nicaragua_weekly_subset.csv"]
EXAMPLE_EVALUATIONS = ["example_evaluation.nc", "example_evaluation_2.nc"]
EXAMPLE_MODEL = Path(__file__).parent.parent.parent / "external_models" / "naive_python_model_uv"


def get_workdir() -> Path:
    """Workdir the UI was started with (`chap ui --workdir`), falling back to ./chap-ui."""
    workdir = Path(os.environ.get("CHAP_UI_WORKDIR", "chap-ui"))
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def example_files(names: list[str]) -> list[Path]:
    """The named files from the repository's example_data directory that exist."""
    return [EXAMPLE_DATA_DIR / name for name in names if (EXAMPLE_DATA_DIR / name).exists()]


def save_upload(workdir: Path, name: str, content: bytes) -> Path:
    """Store an uploaded file under `<workdir>/uploads/` and return its path."""
    path = workdir / "uploads" / Path(name).name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def build_eval_command(
    model_name: str,
    dataset_csv: str,
    output_file: Path,
    backtest_params: BacktestParams,
    model_configuration_yaml: Path | None = None,
) -> list[str]:
    """Arguments for `chap eval`, without the leading executable."""
    args = [
        "eval",
        "--model-name",
        model_name,
        "--dataset-csv",
        dataset_csv,
        "--output-file",
        str(output_file),
        "--backtest-params.n-periods",
        str(backtest_params.n_periods),
        "--backtest-params.n-splits",
        str(backtest_params.n_splits),
        "--backtest-params.stride",
        str(backtest_params.stride),
    ]
    if model_configuration_yaml is not None:
        args += ["--model-configuration-yaml", str(model_configuration_yaml)]
    return args


def format_cli_command(args: list[str]) -> str:
    """The `chap ...` command a user could paste into a terminal."""
    return shlex.join(["chap", *args])


def new_run_dir(workdir: Path, model_name: str) -> Path:
    """A fresh directory under `<workdir>/runs/` named after the time and the model."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", model_name.rstrip("/").split("/")[-1]).strip("-") or "model"
    run_dir = workdir / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}_{slug}"
    run_dir.mkdir(parents=True)
    return run_dir


def start_eval(args: list[str], run_dir: Path) -> subprocess.Popen:
    """Run `chap eval` in the background, logging stdout and stderr to the run directory."""
    (run_dir / COMMAND_NAME).write_text(format_cli_command(args) + "\n")
    log = open(run_dir / LOG_NAME, "w")
    return subprocess.Popen(
        [sys.executable, "-m", "chap_core.cli", *args],
        stdout=log,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


def list_evaluations(workdir: Path) -> list[Path]:
    """Evaluation files produced by UI runs, newest first."""
    return sorted(workdir.glob(f"runs/*/{EVAL_OUTPUT_NAME}"), reverse=True)


def make_plot(nc_path: Path, plot_id: str):
    """Altair chart for one registered backtest plot of an evaluation file."""
    from chap_core.assessment.backtest_plots import create_plot_from_evaluation
    from chap_core.assessment.evaluation import Evaluation

    return create_plot_from_evaluation(plot_id, Evaluation.from_file(nc_path))
