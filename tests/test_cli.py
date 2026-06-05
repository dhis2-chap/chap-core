from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from chap_core.api import forecast
import pytest
from chap_core.util import docker_available
from chap_core.cli_endpoints.evaluate import eval_cmd
from chap_core.cli_endpoints.utils import sanity_check_model
from chap_core.cli_endpoints.utils import test as run_chap_test


@pytest.mark.skipif(not docker_available(), reason="Docker not available")
@pytest.mark.skip(reason="Failing on CI")
def test_forecast_github_model():
    repo_url = "https://github.com/knutdrand/external_rmodel_example.git"
    results = forecast("external", "hydromet_5_filtered", 12, repo_url)


def test_eval_cmd(tmp_path):
    from chap_core.file_io.example_data_set import datasets
    from chap_core.api_types import BacktestParams, RunConfig

    # Export hydromet dataset to CSV for testing
    dataset = datasets["hydromet_5_filtered"].load()
    csv_path = tmp_path / "test_data.csv"
    dataset.to_csv(csv_path)

    # Prepare parameters
    backtest_params = BacktestParams(n_periods=3, n_splits=2, stride=1)
    run_config = RunConfig()

    # Run eval_cmd with CSV
    output_file = tmp_path / "evaluation.nc"
    eval_cmd(
        model_name="https://github.com/dhis2-chap/minimalist_example_lag",
        dataset_csv=csv_path,
        output_file=output_file,
        backtest_params=backtest_params,
        run_config=run_config,
    )

    # Verify output file was created
    assert output_file.exists()


def _make_fake_estimator(min_prediction_length, max_prediction_length):
    from chap_core.database.model_templates_and_config_tables import ModelTemplateInformation

    info = ModelTemplateInformation(
        min_prediction_length=min_prediction_length,
        max_prediction_length=max_prediction_length,
    )
    estimator = MagicMock(name="FakeEstimator")
    estimator.model_information = info
    return estimator


def _patched_eval_chain(fake_estimator, patch_filter=True):
    """Stack mocks for the parts of eval_cmd that aren't under test, returning
    the Evaluation mock so the caller can inspect ``Evaluation.create`` calls.

    By default the pre-backtest region filter is patched to a passthrough so the
    MagicMock dataset survives; tests exercising the filter pass ``patch_filter=False``.
    """
    template_cm = MagicMock(name="ModelTemplate")
    template_cm.__enter__.return_value = template_cm
    template_cm.__exit__.return_value = False
    template_cm.model_template_config.name = "test_model"
    template_cm.model_template_config.version = "1"

    stack = ExitStack()
    stack.enter_context(
        patch(
            "chap_core.cli_endpoints.evaluate.resolve_csv_path",
            return_value=("dummy.csv", None),
        )
    )
    stack.enter_context(patch("chap_core.cli_endpoints.evaluate.discover_geojson", return_value=None))
    stack.enter_context(
        patch(
            "chap_core.cli_endpoints.evaluate.load_dataset_from_csv",
            return_value=MagicMock(name="DataSet"),
        )
    )
    if patch_filter:
        stack.enter_context(
            patch(
                "chap_core.rest_api.db_worker_functions.validate_and_filter_dataset_for_evaluation",
                side_effect=lambda dataset, **_kwargs: dataset,
            )
        )
    stack.enter_context(patch("chap_core.log_config.initialize_logging"))
    mt_mock = stack.enter_context(patch("chap_core.models.model_template.ModelTemplate"))
    mt_mock.from_directory_or_github_url.return_value = template_cm
    stack.enter_context(
        patch(
            "chap_core.cli_endpoints.evaluate.get_configuration",
            return_value=None,
        )
    )
    stack.enter_context(
        patch(
            "chap_core.cli_endpoints.evaluate.get_estimator",
            return_value=fake_estimator,
        )
    )
    eval_mock = stack.enter_context(patch("chap_core.assessment.evaluation.Evaluation"))
    eval_mock.create.return_value = MagicMock(name="EvaluationInstance")
    return stack, eval_mock


def test_eval_cmd_raises_when_n_periods_below_min_prediction_length(tmp_path):
    """Dispatch in evaluate.py guards against horizons shorter than the model's declared min."""
    from chap_core.api_types import BacktestParams, RunConfig

    fake_estimator = _make_fake_estimator(min_prediction_length=5, max_prediction_length=10)
    stack, _ = _patched_eval_chain(fake_estimator)
    with stack:
        with pytest.raises(ValueError, match="minimum prediction length"):
            eval_cmd(
                model_name="dummy",
                dataset_csv="dummy.csv",
                output_file=tmp_path / "out.nc",
                backtest_params=BacktestParams(n_periods=3, n_splits=2, stride=1),
                run_config=RunConfig(),
            )


def test_eval_cmd_wraps_in_extended_predictor_when_n_periods_above_max(tmp_path):
    """When the requested horizon exceeds the model's declared max, the dispatch wraps
    the estimator in ExtendedPredictor and forwards the wrapped estimator to evaluation."""
    from chap_core.api_types import BacktestParams, RunConfig
    from chap_core.external.ExtendedPredictor import ExtendedPredictor

    fake_estimator = _make_fake_estimator(min_prediction_length=1, max_prediction_length=2)
    stack, eval_mock = _patched_eval_chain(fake_estimator)
    with stack:
        eval_cmd(
            model_name="dummy",
            dataset_csv="dummy.csv",
            output_file=tmp_path / "out.nc",
            backtest_params=BacktestParams(n_periods=5, n_splits=2, stride=1),
            run_config=RunConfig(),
        )

    assert eval_mock.create.call_count == 1
    forwarded = eval_mock.create.call_args.kwargs["estimator"]
    assert isinstance(forwarded, ExtendedPredictor)


def test_eval_cmd_wraps_when_only_max_set_and_below_n_periods(tmp_path):
    """Real models often declare max_prediction_length but leave min unset
    (e.g. chap-models/Vietnam-dengue-superensemble declares max=1, no min).
    The dispatch must still honour the declared max even when min is None."""
    from chap_core.api_types import BacktestParams, RunConfig
    from chap_core.external.ExtendedPredictor import ExtendedPredictor

    fake_estimator = _make_fake_estimator(min_prediction_length=None, max_prediction_length=2)
    stack, eval_mock = _patched_eval_chain(fake_estimator)
    with stack:
        eval_cmd(
            model_name="dummy",
            dataset_csv="dummy.csv",
            output_file=tmp_path / "out.nc",
            backtest_params=BacktestParams(n_periods=5, n_splits=2, stride=1),
            run_config=RunConfig(),
        )

    assert eval_mock.create.call_count == 1
    forwarded = eval_mock.create.call_args.kwargs["estimator"]
    assert isinstance(forwarded, ExtendedPredictor)


def test_eval_cmd_raises_when_only_min_set_and_above_n_periods(tmp_path):
    """The min-bound check must also fire when only min is declared and max is None."""
    from chap_core.api_types import BacktestParams, RunConfig

    fake_estimator = _make_fake_estimator(min_prediction_length=5, max_prediction_length=None)
    stack, _ = _patched_eval_chain(fake_estimator)
    with stack:
        with pytest.raises(ValueError, match="minimum prediction length"):
            eval_cmd(
                model_name="dummy",
                dataset_csv="dummy.csv",
                output_file=tmp_path / "out.nc",
                backtest_params=BacktestParams(n_periods=3, n_splits=2, stride=1),
                run_config=RunConfig(),
            )


def test_eval_cmd_does_not_wrap_when_bounds_unspecified(tmp_path):
    """When the model declares neither min nor max, dispatch logs a warning and forwards
    the original estimator unchanged."""
    from chap_core.api_types import BacktestParams, RunConfig

    fake_estimator = _make_fake_estimator(min_prediction_length=None, max_prediction_length=None)
    stack, eval_mock = _patched_eval_chain(fake_estimator)
    with stack:
        eval_cmd(
            model_name="dummy",
            dataset_csv="dummy.csv",
            output_file=tmp_path / "out.nc",
            backtest_params=BacktestParams(n_periods=3, n_splits=2, stride=1),
            run_config=RunConfig(),
        )

    assert eval_mock.create.call_count == 1
    forwarded = eval_mock.create.call_args.kwargs["estimator"]
    assert forwarded is fake_estimator


def test_eval_cmd_drops_regions_with_no_disease_cases(tmp_path, caplog):
    """Regions whose entire training-period disease_cases is NaN must be dropped before
    the backtest runs, matching the REST API's pre-backtest filtering. The names of the
    dropped regions must also be surfaced in the warning log."""
    import logging

    import numpy as np

    from chap_core.api_types import BacktestParams, RunConfig
    from chap_core.datatypes import HealthData
    from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
    from chap_core.time_period import PeriodRange

    period_range = PeriodRange.from_strings([f"2023-{m:02d}" for m in range(1, 13)])
    valid = HealthData(time_period=period_range, disease_cases=np.arange(1, 13, dtype=float))
    nan_region = HealthData(time_period=period_range, disease_cases=np.full(12, np.nan))
    dataset = DataSet({"valid_region": valid, "nan_region": nan_region})

    fake_estimator = _make_fake_estimator(min_prediction_length=None, max_prediction_length=None)
    stack, eval_mock = _patched_eval_chain(fake_estimator, patch_filter=False)
    stack.enter_context(patch("chap_core.cli_endpoints.evaluate.load_dataset_from_csv", return_value=dataset))
    with stack, caplog.at_level(logging.WARNING, logger="chap_core.rest_api.db_worker_functions"):
        eval_cmd(
            model_name="dummy",
            dataset_csv="dummy.csv",
            output_file=tmp_path / "out.nc",
            backtest_params=BacktestParams(n_periods=3, n_splits=2, stride=1),
            run_config=RunConfig(),
        )

    forwarded = eval_mock.create.call_args.kwargs["dataset"]
    locations = list(forwarded.locations())
    assert "nan_region" not in locations
    assert "valid_region" in locations

    rejection_warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "Rejected regions" in r.message]
    assert len(rejection_warnings) == 1
    assert "nan_region" in rejection_warnings[0].message


def test_eval_cmd_track_logs_params_metrics_and_artifacts(tmp_path, monkeypatch):
    """With run_config.track, params (incl. model config), metrics, and the .nc artifact end up in MLflow."""
    from pathlib import Path

    import mlflow
    import yaml
    from chap_core.api_types import BacktestParams, RunConfig

    tracking_dir = tmp_path / "mlruns"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tracking_dir}")
    mlflow.set_tracking_uri(f"file://{tracking_dir}")

    output_file = tmp_path / "out.nc"

    model_config_yaml = tmp_path / "model_config.yaml"
    model_config_yaml.write_text(
        yaml.safe_dump({"learning_rate": 0.01, "n_estimators": 100, "optimizer": {"name": "adam", "momentum": 0.9}})
    )

    def _write_nc(filepath, **_kwargs):
        Path(filepath).write_bytes(b"netcdf-placeholder")

    fake_estimator = _make_fake_estimator(min_prediction_length=None, max_prediction_length=None)
    stack, eval_mock = _patched_eval_chain(fake_estimator)
    eval_instance = MagicMock(name="EvaluationInstance")
    eval_instance.to_file.side_effect = _write_nc
    eval_mock.create.return_value = eval_instance

    metrics_stub = patch(
        "chap_core.assessment.eval_tracking.calculate_metrics",
        return_value={"mae": 1.23, "rmse": 4.56, "non_numeric": None},
    )

    with stack, metrics_stub:
        eval_cmd(
            model_name="dummy-model",
            dataset_csv="dummy.csv",
            output_file=output_file,
            backtest_params=BacktestParams(n_periods=3, n_splits=2, stride=1),
            run_config=RunConfig(track=True),
            model_configuration_yaml=model_config_yaml,
        )

    runs = mlflow.search_runs(experiment_names=["chap-backtests"], output_format="list")
    assert len(runs) == 1
    run = runs[0]

    assert run.data.params["model_name"] == "dummy-model"
    assert run.data.params["dataset_csv"] == "dummy.csv"
    assert run.data.params["n_periods"] == "3"
    assert run.data.params["n_splits"] == "2"
    assert run.data.params["stride"] == "1"
    assert run.data.params["prediction_length"] == "3"

    assert run.data.params["model.learning_rate"] == "0.01"
    assert run.data.params["model.n_estimators"] == "100"
    assert run.data.params["model.optimizer.name"] == "adam"
    assert run.data.params["model.optimizer.momentum"] == "0.9"

    assert run.data.metrics["mae"] == 1.23
    assert run.data.metrics["rmse"] == 4.56
    assert "non_numeric" not in run.data.metrics

    client = mlflow.MlflowClient(tracking_uri=f"file://{tracking_dir}")
    artifacts = {a.path for a in client.list_artifacts(run.info.run_id)}  # type: ignore[union-attr]
    assert "out.nc" in artifacts
    assert "model_configuration.json" in artifacts


def test_eval_cmd_track_without_tracking_uri_raises(tmp_path, monkeypatch):
    """run_config.track=True without MLFLOW_TRACKING_URI raises before any heavy work."""
    from chap_core.api_types import BacktestParams, RunConfig
    from chap_core.assessment.eval_tracking import TrackingConfigError

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    fake_estimator = _make_fake_estimator(min_prediction_length=None, max_prediction_length=None)
    stack, _ = _patched_eval_chain(fake_estimator)

    with stack, pytest.raises(TrackingConfigError, match="MLFLOW_TRACKING_URI"):
        eval_cmd(
            model_name="dummy",
            dataset_csv="dummy.csv",
            output_file=tmp_path / "out.nc",
            backtest_params=BacktestParams(n_periods=3, n_splits=2, stride=1),
            run_config=RunConfig(track=True),
        )


def test_eval_cmd_with_data_source_mapping(tmp_path):
    import json

    import pandas as pd

    from chap_core.api_types import BacktestParams, RunConfig
    from chap_core.file_io.example_data_set import datasets

    # Export hydromet dataset to CSV for testing
    dataset = datasets["hydromet_5_filtered"].load()
    csv_path = tmp_path / "test_data.csv"
    dataset.to_csv(csv_path)

    # Rename columns to simulate a CSV with different column names
    df = pd.read_csv(csv_path)
    df.rename(columns={"rainfall": "precip", "disease_cases": "cases"}, inplace=True)
    renamed_csv_path = tmp_path / "renamed_data.csv"
    df.to_csv(renamed_csv_path, index=False)

    # Create mapping JSON file
    mapping = {"rainfall": "precip", "disease_cases": "cases"}
    mapping_path = tmp_path / "column_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f)

    # Prepare parameters
    backtest_params = BacktestParams(n_periods=3, n_splits=2, stride=1)
    run_config = RunConfig()

    # Run eval_cmd with the mapping
    output_file = tmp_path / "evaluation.nc"
    eval_cmd(
        model_name="https://github.com/dhis2-chap/minimalist_example_lag",
        dataset_csv=renamed_csv_path,
        output_file=output_file,
        backtest_params=backtest_params,
        run_config=run_config,
        data_source_mapping=mapping_path,
    )

    # Verify output file was created
    assert output_file.exists()


def test_chap_test_reports_environment_and_passes(capsys):
    """`chap test` prints version/environment, passes its import checks, and exits 0."""
    run_chap_test()

    out = capsys.readouterr().out
    assert "chap-core" in out
    assert "numpy / pandas / xarray ... OK" in out
    assert "public API (data, fetch, ModelTemplateInterface) ... OK" in out
    assert "All checks passed." in out


def test_chap_test_exits_nonzero_when_a_core_import_fails(monkeypatch, capsys):
    """A failing core import is surfaced and makes `chap test` exit non-zero."""
    import importlib

    real_import_module = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "xarray":
            raise ImportError("simulated missing xarray")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(SystemExit) as exc_info:
        run_chap_test()

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "missing: xarray" in out
    assert "Some checks FAILED." in out
