"""MLflow tracking for the ``chap eval`` command.

Tracking is opt-in via ``chap eval --track``. The MLflow tracking URI must be
set explicitly via the ``MLFLOW_TRACKING_URI`` environment variable; this
module never picks a default location, to avoid surprising the user with files
written outside their workspace.
"""

from __future__ import annotations

import hashlib
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import mlflow

import chap_core
from chap_core.assessment.metrics import available_metrics, calculate_metrics

if TYPE_CHECKING:
    from collections.abc import Iterator

    from chap_core.api_types import BacktestParams

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_NAME = "chap-backtests"


class TrackingConfigError(RuntimeError):
    """Raised when --track is requested but MLFLOW_TRACKING_URI is not set."""


class Tracker:
    """Yielded by :func:`tracked_eval_run`. When tracking is disabled every
    method is a no-op, which lets the caller treat tracked and untracked runs
    uniformly without an ``if track`` branch.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def log_outputs_from_files(
        self,
        nc_path: Path,
        plot_path: Path | None = None,
    ) -> None:
        """Log the NetCDF (and optional plot) as artifacts and log metrics
        computed by re-loading the NetCDF. Called after the eval has finished
        writing its outputs to disk."""
        if not self.enabled:
            return
        _log_outputs_from_files(nc_path, plot_path)


@contextmanager
def tracked_eval_run(
    *,
    track: bool,
    model_name: str,
    dataset_csv: str,
    backtest_params: BacktestParams,
    historical_context_years: int,
) -> Iterator[Tracker]:
    """Context manager that opens an MLflow run when ``track`` is True.

    When ``track`` is False this is a no-op. When True it requires
    ``MLFLOW_TRACKING_URI`` to be set, sets the experiment (honoring
    ``MLFLOW_EXPERIMENT_NAME`` if set, else ``"chap-backtests"``), and logs the
    run-identifying params and tags before yielding.
    """
    if not track:
        yield Tracker(enabled=False)
        return

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise TrackingConfigError(
            "chap eval --track requires MLFLOW_TRACKING_URI to be set "
            "(e.g. export MLFLOW_TRACKING_URI=file://$HOME/chap-evaluations/mlruns)"
        )

    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)
    mlflow.set_experiment(experiment_name)

    run_name = f"{_short_name(model_name)}-{_short_name(dataset_csv)}"
    with mlflow.start_run(run_name=run_name):
        params: dict[str, object] = {
            "model_name": model_name,
            "dataset_csv": dataset_csv,
            "n_periods": backtest_params.n_periods,
            "n_splits": backtest_params.n_splits,
            "stride": backtest_params.stride,
            "prediction_length": backtest_params.n_periods,
            "historical_context_years": historical_context_years,
            "chap_version": chap_core.__version__,
        }
        sha = _maybe_file_sha256(dataset_csv)
        if sha is not None:
            params["dataset_sha256"] = sha
        mlflow.log_params(params)

        tags: dict[str, str] = {
            "dataset.name": _short_name(dataset_csv),
            "model.name": _short_name(model_name),
        }
        git_sha = _chap_core_git_sha()
        if git_sha is not None:
            tags["git_sha"] = git_sha
        mlflow.set_tags(tags)

        yield Tracker(enabled=True)


def _log_outputs_from_files(nc_path: Path, plot_path: Path | None) -> None:
    if not nc_path.exists():
        logger.warning("No NetCDF at %s; skipping artifact + metric logging", nc_path)
        return

    mlflow.log_artifact(str(nc_path))
    if plot_path is not None and plot_path.exists():
        mlflow.log_artifact(str(plot_path))

    from chap_core.assessment.evaluation import Evaluation

    try:
        evaluation = Evaluation.from_file(nc_path)
        raw = calculate_metrics(evaluation, list(available_metrics.keys()))
    except Exception:
        logger.exception("Failed to compute metrics for MLflow tracking; skipping log_metrics")
        return

    numeric = {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}
    if numeric:
        mlflow.log_metrics(numeric)
    skipped = sorted(set(raw) - set(numeric))
    if skipped:
        logger.info("Skipped non-numeric / inapplicable metrics: %s", ", ".join(skipped))


def _short_name(value: str) -> str:
    """Derive a short display name from a path, URL, or repo identifier."""
    parsed = urlparse(value)
    candidate = parsed.path if parsed.scheme else value
    base = Path(candidate).name or candidate
    return Path(base).stem or base


def _maybe_file_sha256(dataset_csv: str) -> str | None:
    parsed = urlparse(dataset_csv)
    if parsed.scheme in {"http", "https", "s3", "gs"}:
        return None
    path = Path(dataset_csv)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _chap_core_git_sha() -> str | None:
    """Return the short SHA of the chap-core repo if chap-core is installed
    from a git checkout, else None."""
    try:
        import git
    except ImportError:
        return None
    pkg_dir = Path(chap_core.__file__).resolve().parent
    try:
        repo = git.Repo(pkg_dir, search_parent_directories=True)
        return repo.head.commit.hexsha[:12]
    except Exception:
        return None
