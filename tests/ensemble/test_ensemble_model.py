import pytest

from chap_core.ensemble.ensemble_model import EnsembleModel


def test_probabilistic_disallows_residual_bootstrap():
    with pytest.raises(ValueError, match="Residual bootstrap is only supported for deterministic ensembles"):
        EnsembleModel(
            base_templates=[object()],
            method="probabilistic",
            use_residual_bootstrap=True,
        )
