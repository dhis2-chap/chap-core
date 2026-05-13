import pytest

from chap_core.ensemble.ensemble_model import EnsembleModel
from chap_core.datatypes import FullData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


def test_probabilistic_disallows_residual_bootstrap():
    with pytest.raises(ValueError, match="Residual bootstrap is only supported for deterministic ensembles"):
        EnsembleModel(
            base_templates=[object()],
            method="probabilistic",
            use_residual_bootstrap=True,
        )


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
