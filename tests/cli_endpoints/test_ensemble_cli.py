import numpy as np

from chap_core.api_types import BackTestParams, RunConfig
from chap_core.cli_endpoints import ensemble as ensemble_cli
from chap_core.datatypes import Samples
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


def test_evaluate_ensemble_passes_residual_bootstrap(monkeypatch):
    captured: dict[str, object] = {}

    def fake_core(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(ensemble_cli, "_evaluate_ensemble_core", fake_core)

    ensemble_cli.evaluate_ensemble(
        base_model_names="model_a",
        ensemble_method="probabilistic",
        use_residual_bootstrap=True,
    )

    assert captured["use_residual_bootstrap"] is True


def test_evaluate_ensemble_smoke(weekly_full_data, tmp_path, monkeypatch):
    def fake_load_dataset(**_kwargs):
        return weekly_full_data

    monkeypatch.setattr(ensemble_cli, "_load_dataset", fake_load_dataset)

    class _DummyTemplate:
        def __init__(self, name: str, value: float):
            self.name = name
            self._value = value

        def get_model(self, _config):
            return lambda: _ConstantEstimator(self._value, 1)

    class _ConstantPredictor:
        def __init__(self, value: float, n_samples: int):
            self._value = value
            self._n_samples = n_samples

        def predict(self, _historic_data, future_data):
            result = {}
            for loc in future_data.locations():
                tp = future_data[loc].time_period
                vals = np.full(len(tp), self._value, dtype=float)
                samples = np.tile(vals.reshape(-1, 1), (1, self._n_samples))
                result[loc] = Samples(tp, samples)
            return DataSet(result)

    class _ConstantEstimator:
        def __init__(self, value: float, n_samples: int):
            self._value = value
            self._n_samples = n_samples

        def train(self, _train_data):
            return _ConstantPredictor(self._value, self._n_samples)

    def fake_from_directory_or_github_url(cls, name, **_kwargs):
        value = 2.0 if "b" in name else 1.0
        return _DummyTemplate(name, value)

    monkeypatch.setattr(
        ensemble_cli.ModelTemplate,
        "from_directory_or_github_url",
        classmethod(fake_from_directory_or_github_url),
    )

    report_path = tmp_path / "ensemble_report.csv"
    results = ensemble_cli.evaluate_ensemble(
        base_model_names="model_a,model_b",
        ensemble_method="deterministic",
        dataset_name=None,
        dataset_country=None,
        dataset_csv=None,
        polygons_json=None,
        polygons_id_field="id",
        report_filename=report_path,
        output_file=None,
        backtest_params=BackTestParams(n_periods=1, n_splits=1, stride=1),
        run_config=RunConfig(),
        model_configuration_yaml=None,
        random_state=123,
        use_residual_bootstrap=False,
        data_source_mapping=None,
        historical_context_years=1,
    )

    assert results
    assert report_path.with_suffix(".csv").exists()
