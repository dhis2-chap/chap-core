import numpy as np
import pytest

from chap_core.api_types import BacktestParams, RunConfig
from chap_core.cli_endpoints import ensemble as ensemble_cli
from chap_core.datatypes import Samples
from chap_core.external.model_configuration import ModelTemplateConfigV2
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


class _ConstantModel:
    """Stands in for an ExternalModel: self-returning when called, trains to itself."""

    def __init__(self, value: float, config: ModelTemplateConfigV2, seen_period_counts: list[int]):
        self._value = value
        self._config = config
        self.seen_period_counts = seen_period_counts

    @property
    def model_information(self) -> ModelTemplateConfigV2:
        return self._config

    def __call__(self) -> "_ConstantModel":
        return self

    def train(self, _train_data, extra_args=None) -> "_ConstantModel":
        return self

    def predict(self, _historic_data, future_data) -> DataSet:
        self.seen_period_counts.append(len(future_data.period_range))
        result = {}
        for loc in future_data.locations():
            tp = future_data[loc].time_period
            vals = np.full(len(tp), self._value, dtype=float)
            result[loc] = Samples(tp, vals.reshape(-1, 1))
        return DataSet(result)


class _DummyTemplate:
    def __init__(self, name: str, value: float, config: ModelTemplateConfigV2):
        self.name = name
        self._value = value
        self._config = config
        self.seen_period_counts: list[int] = []
        self.entered = False
        self.exited = False

    @property
    def model_template_config(self) -> ModelTemplateConfigV2:
        return self._config

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
        return _ConstantModel(self._value, self._config, self.seen_period_counts)


def _install_templates(monkeypatch, weekly_full_data, config_for_name) -> list[_DummyTemplate]:
    """Patch dataset loading and template resolution, returning the templates created."""

    def fake_load_dataset(**_kwargs):
        return weekly_full_data

    monkeypatch.setattr(ensemble_cli, "_load_dataset", fake_load_dataset)

    created: list[_DummyTemplate] = []

    def fake_from_directory_or_github_url(cls, name, **_kwargs):
        value = 2.0 if "b" in name else 1.0
        template = _DummyTemplate(name, value, config_for_name(name))
        created.append(template)
        return template

    from chap_core.models.model_template import ModelTemplate

    monkeypatch.setattr(
        ModelTemplate,
        "from_directory_or_github_url",
        classmethod(fake_from_directory_or_github_url),
    )
    return created


def _run(report_path, **overrides):
    kwargs = dict(
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
    kwargs.update(overrides)
    return ensemble_cli.evaluate_ensemble(**kwargs)


def test_evaluate_ensemble_smoke(weekly_full_data, tmp_path, monkeypatch):
    created = _install_templates(
        monkeypatch,
        weekly_full_data,
        lambda name: ModelTemplateConfigV2(name=name, min_prediction_length=1, max_prediction_length=4),
    )

    report_path = tmp_path / "ensemble_report.csv"
    results = _run(report_path)

    assert results
    assert report_path.with_suffix(".csv").exists()
    assert created, "no templates were loaded"
    assert all(t.entered and t.exited for t in created)


def test_meta_report_follows_report_stem_and_keeps_coefficients(weekly_full_data, tmp_path, monkeypatch):
    """A fixed meta report name let two runs in one directory clobber each other's weights."""
    _install_templates(
        monkeypatch,
        weekly_full_data,
        lambda name: ModelTemplateConfigV2(name=name, min_prediction_length=1, max_prediction_length=4),
    )

    report_path = tmp_path / "run_one.csv"
    _run(report_path)

    meta_path = tmp_path / "run_one_meta.csv"
    assert meta_path.exists()
    lines = meta_path.read_text(encoding="utf-8").strip().split("\n")
    assert lines[0] == "Model,round,quantity,model_a,model_b"
    quantities = [line.split(",")[2] for line in lines[1:]]
    # The deterministic meta-model applies the raw coefficients, not the normalised
    # shares, so both have to be reported.
    assert quantities == ["weight_percent", "coefficient"]


def test_base_model_below_minimum_prediction_length_is_rejected(weekly_full_data, tmp_path, monkeypatch):
    """`chap evaluate` refuses this; evaluate-ensemble used to ask for it anyway."""
    _install_templates(
        monkeypatch,
        weekly_full_data,
        lambda name: ModelTemplateConfigV2(name=name, min_prediction_length=3, max_prediction_length=6),
    )

    with pytest.raises(ValueError, match="minimum prediction length"):
        _run(tmp_path / "ensemble_report.csv", backtest_params=BacktestParams(n_periods=2, n_splits=1, stride=1))


def test_base_model_above_maximum_prediction_length_is_extended(weekly_full_data, tmp_path, monkeypatch):
    """A capped base model must be iterated, not silently asked for the full horizon."""
    created = _install_templates(
        monkeypatch,
        weekly_full_data,
        lambda name: ModelTemplateConfigV2(name=name, min_prediction_length=1, max_prediction_length=1),
    )

    _run(
        tmp_path / "ensemble_report.csv",
        backtest_params=BacktestParams(n_periods=2, n_splits=1, stride=1),
        inner_val_periods=4,
    )

    assert created
    for template in created:
        assert template.seen_period_counts, "base model was never asked to predict"
        assert max(template.seen_period_counts) == 1
