"""Unit tests for the eLoss faithfulness metric."""

import numpy as np
import pandas as pd
import pytest

from chap_core.explainability import lime as lime_module
from chap_core.explainability.testing.metrics import eLoss


def _common_args() -> dict:
    """Minimal kwargs to call eLoss with; the test patches the heavy parts."""
    return dict(
        model=None,  # type: ignore[arg-type]
        original_vector={},
        feature_map=[],
        sampler=None,  # type: ignore[arg-type]
        hist_df=pd.DataFrame(),
        fut_df=pd.DataFrame(),
        features_hist=[],
        features_fut=[],
        horizon=1,
        location="loc",
        hist_type=None,
        fut_type=None,
        feat_indices={},
        y_orig=0.0,
        full_dataset=None,
        full_future_weather=None,
    )


class TestEarlyReturns:
    def test_empty_feature_names_returns_zeros(self):
        delta, top, bottom = eLoss(
            **_common_args(),
            feature_names=[],
            sorted_explanation=[("anything", 1.0)],
        )
        assert (delta, top, bottom) == (0.0, 0.0, 0.0)

    def test_no_overlap_between_explanation_and_feature_names_returns_zeros(self):
        delta, top, bottom = eLoss(
            **_common_args(),
            feature_names=["a", "b"],
            sorted_explanation=[("c", 1.0), ("d", 0.5)],
        )
        assert (delta, top, bottom) == (0.0, 0.0, 0.0)


class TestFaithfulnessSign:
    """If the explanation is faithful (top-k perturbations move the model more
    than bottom-k), delta_eloss should come out positive. Patch the deviation
    machinery so the test is deterministic and fast."""

    def test_faithful_explanation_yields_positive_delta(self, monkeypatch):
        feature_names = [f"f{i}" for i in range(10)]
        # Sorted so that f0 is most important, f9 least.
        sorted_explanation = [(name, 10.0 - i) for i, name in enumerate(feature_names)]

        # Fake perturb_vectors: pass masks straight through so we can read them
        # in the fake produce_lime_dataset.
        def fake_perturb(*args, **kwargs):
            # args[5] = flat_masks (the 6th positional arg in real signature),
            # but the metrics module calls perturb_vectors(hist_df, original,
            # feat_indices, sampler, feature_map, masks) -> positional.
            masks = args[5]
            return masks, masks  # (perturbations, perturbation_masks)

        # Fake produce_lime_dataset: deviation grows with the number of
        # "important" indices turned off. We say indices 0..4 are important
        # (high weight) and 5..9 are noise (zero weight) — so a faithful
        # explanation (which orders f0..f9 in that exact order) should
        # produce a *larger* response when its top-k features are off than
        # when its bottom-k are.
        importance_weights = np.array([1.0] * 5 + [0.0] * 5)

        def fake_produce(model, hist_df, future_df, perturbations, *_args, **_kwargs):
            ys = []
            for mask in perturbations:
                # |y_perturbed - y_orig| modelled as sum of importance_weight
                # at every "off" position.
                off_indices = mask == 0
                deviation = float(np.sum(importance_weights * off_indices))
                ys.append(deviation)
            return None, np.asarray(ys), None, None

        monkeypatch.setattr(lime_module, "perturb_vectors", fake_perturb)
        monkeypatch.setattr(lime_module, "produce_lime_dataset", fake_produce)

        delta, auc_top, auc_bottom = eLoss(
            **_common_args(),
            feature_names=feature_names,
            sorted_explanation=sorted_explanation,
        )

        assert auc_top > auc_bottom
        assert delta > 0
        assert np.isfinite(delta)

    def test_anti_faithful_explanation_yields_negative_delta(self, monkeypatch):
        """If the explanation ranks the genuinely-important features last,
        delta should flip sign — perturbing 'top-k' (which are actually
        irrelevant) barely moves the model, while perturbing 'bottom-k'
        (which are actually important) moves it a lot."""
        feature_names = [f"f{i}" for i in range(10)]
        # Reversed ranking: f9 marked most important even though indices 0..4
        # are the genuinely-important ones.
        sorted_explanation = [(name, 1.0 + i) for i, name in enumerate(reversed(feature_names))]

        def fake_perturb(*args, **_kwargs):
            masks = args[5]
            return masks, masks

        importance_weights = np.array([1.0] * 5 + [0.0] * 5)

        def fake_produce(model, hist_df, future_df, perturbations, *_args, **_kwargs):
            ys = [float(np.sum(importance_weights * (m == 0))) for m in perturbations]
            return None, np.asarray(ys), None, None

        monkeypatch.setattr(lime_module, "perturb_vectors", fake_perturb)
        monkeypatch.setattr(lime_module, "produce_lime_dataset", fake_produce)

        delta, _, _ = eLoss(
            **_common_args(),
            feature_names=feature_names,
            sorted_explanation=sorted_explanation,
        )

        assert delta < 0


class TestReturnShape:
    def test_returns_three_finite_floats(self, monkeypatch):
        feature_names = ["a", "b", "c", "d"]
        sorted_explanation = [("a", 0.4), ("b", 0.3), ("c", 0.2), ("d", 0.1)]

        def fake_perturb(*args, **_kwargs):
            masks = args[5]
            return masks, masks

        def fake_produce(*_args, **_kwargs):
            # produce a deviation curve of length matching #masks (any value)
            perturbations = _args[3]
            return None, np.full(len(perturbations), 0.5), None, None

        monkeypatch.setattr(lime_module, "perturb_vectors", fake_perturb)
        monkeypatch.setattr(lime_module, "produce_lime_dataset", fake_produce)

        result = eLoss(
            **_common_args(),
            feature_names=feature_names,
            sorted_explanation=sorted_explanation,
        )
        assert len(result) == 3
        assert all(isinstance(v, float) and np.isfinite(v) for v in result)


@pytest.mark.parametrize("n_buckets", [1, 5, 20])
def test_n_buckets_parameter_does_not_crash(n_buckets, monkeypatch):
    feature_names = ["a", "b", "c"]
    sorted_explanation = [("a", 1.0), ("b", 0.5), ("c", 0.1)]

    def fake_perturb(*args, **_kwargs):
        masks = args[5]
        return masks, masks

    def fake_produce(*_args, **_kwargs):
        perturbations = _args[3]
        return None, np.zeros(len(perturbations)), None, None

    monkeypatch.setattr(lime_module, "perturb_vectors", fake_perturb)
    monkeypatch.setattr(lime_module, "produce_lime_dataset", fake_produce)

    delta, _, _ = eLoss(
        **_common_args(),
        feature_names=feature_names,
        sorted_explanation=sorted_explanation,
        n_buckets=n_buckets,
    )
    assert delta == 0.0  # constant deviation -> top and bottom AUCs cancel out
