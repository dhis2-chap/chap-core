import pytest

from chap_core.ensemble.ensemble_model import EnsembleModel
from chap_core.datatypes import FullData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


def test_train_masks_nan_features(weekly_full_data, constant_template_factory, nan_template_factory):
    templates = [
        constant_template_factory(1.0, 1, "model_a"),
        nan_template_factory(2.0, 1, "model_nan"),
    ]
    model = EnsembleModel(base_templates=templates, method="deterministic", n_samples=2)

    predictor = model.train(weekly_full_data)

    assert predictor is not None
    assert model.weights is not None


def test_requires_base_templates():
    with pytest.raises(ValueError, match="Need at least one base model"):
        EnsembleModel(base_templates=[])


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="invalid"):
        EnsembleModel(base_templates=[object()], method="invalid")


def test_train_requires_two_periods(weekly_full_data, constant_template_factory):
    df = weekly_full_data.to_pandas()
    first_period = df["time_period"].iloc[0]
    df_one = df[df["time_period"] == first_period].copy()
    one_period = DataSet.from_pandas(df_one, FullData)

    templates = [constant_template_factory(1.0, 1, "model_a")]
    model = EnsembleModel(base_templates=templates, method="deterministic")

    with pytest.raises(ValueError, match="Need at least two time periods"):
        model.train(one_period)


def test_train_invalid_split_raises(weekly_full_data, constant_template_factory):
    templates = [constant_template_factory(1.0, 1, "model_a")]
    model = EnsembleModel(base_templates=templates, method="deterministic", inner_val_periods=0)

    with pytest.raises(ValueError, match="Invalid inner validation split"):
        model.train(weekly_full_data)


def test_inner_validation_windows_match_horizon(weekly_full_data, constant_template_factory):
    """Weights must be fitted at the horizon the ensemble is actually used at."""
    templates = [constant_template_factory(1.0, 1, "model_a")]
    model = EnsembleModel(base_templates=templates, method="deterministic", inner_val_periods=6, horizon=3)

    windows = model.inner_validation_windows(weekly_full_data)

    assert len(windows) == 2
    for _historic, future in windows:
        assert len(list(future.period_range)) == 3


def test_inner_validation_masks_target(weekly_full_data, recording_template_factory):
    """The target must never reach a base model's future_data."""
    templates = [recording_template_factory(1.0, 1, "model_a")]
    model = EnsembleModel(base_templates=templates, method="deterministic", inner_val_periods=4, horizon=2)

    model.train(weekly_full_data)

    seen = templates[0].seen_future_fields
    assert seen, "base model was never asked to predict"
    for fields in seen:
        assert "disease_cases" not in fields


def test_inner_validation_drops_partial_trailing_window(weekly_full_data, constant_template_factory):
    """A short trailing window would score base models at horizons 1..k and mix those
    rows into the same weight fit as the full-horizon rows."""
    templates = [constant_template_factory(1.0, 1, "model_a")]
    model = EnsembleModel(base_templates=templates, method="deterministic", inner_val_periods=10, horizon=4)

    windows = model.inner_validation_windows(weekly_full_data)

    assert len(windows) == 2
    for _historic, future in windows:
        assert len(list(future.period_range)) == 4


def test_inner_validation_requires_one_full_window(weekly_full_data, constant_template_factory):
    templates = [constant_template_factory(1.0, 1, "model_a")]
    model = EnsembleModel(base_templates=templates, method="deterministic", inner_val_periods=2, horizon=5)

    with pytest.raises(ValueError, match="at least 5 periods"):
        model.inner_validation_windows(weekly_full_data)
