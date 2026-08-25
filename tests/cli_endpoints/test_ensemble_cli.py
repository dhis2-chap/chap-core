import numpy as np

from chap_core.api_types import BacktestParams, RunConfig
from chap_core.cli_endpoints import ensemble as ensemble_cli
from chap_core.datatypes import Samples
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


def test_evaluate_ensemble_smoke(weekly_full_data, tmp_path, monkeypatch):
    def fake_load_dataset(**_kwargs):
        return weekly_full_data

    monkeypatch.setattr(ensemble_cli, "_load_dataset", fake_load_dataset)

    class _DummyTemplate:
        def __init__(self, name: str, value: float):
            self.name = name
            self._value = value
            self.entered = False
            self.exited = False

        def __enter__(self):
            # The CLI must enter the template: for chapkit models this is what starts
            # the backing service, and skipping it left get_model raising at runtime.
            self.entered = True
            return self

        def __exit__(self, *_exc):
            self.exited = True
            return False

        def get_model(self, _config):
            assert self.entered, "get_model called before the template was entered"
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

    created: list[_DummyTemplate] = []

    def fake_from_directory_or_github_url(cls, name, **_kwargs):
        value = 2.0 if "b" in name else 1.0
        template = _DummyTemplate(name, value)
        created.append(template)
        return template

    from chap_core.models.model_template import ModelTemplate

    monkeypatch.setattr(
        ModelTemplate,
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
        backtest_params=BacktestParams(n_periods=1, n_splits=1, stride=1),
        run_config=RunConfig(),
        model_configuration_yaml=None,
        inner_val_periods=4,
        data_source_mapping=None,
        historical_context_years=1,
    )

    assert results
    assert report_path.with_suffix(".csv").exists()
    assert created, "no templates were loaded"
    assert all(t.entered and t.exited for t in created)
