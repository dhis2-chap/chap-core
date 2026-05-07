import pytest

from chap_core.ensemble.ensemble_model import EnsembleModel


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
