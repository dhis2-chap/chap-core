"""End-to-end integration tests for ``explain()`` against a mock ExternalModel.

The whole point of these tests is to exercise the LIME orchestration in
``lime.py`` (segmentation → perturbation → predict → surrogate fit →
coefficient extraction) without needing a trained model directory under
``runs/``. The mock model satisfies the ``ExternalModel.predict()``
contract — taking historic + future DataSets and returning a
``DataSet[Samples]`` — so the real pipeline runs unmodified against it.

These would have caught the pre-existing ``log1p`` NaN crash on their own.
Keep them as the floor for future explainability changes.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pytest

from chap_core.datatypes import FullData, Samples
from chap_core.explainability.lime import explain, explain_adaptive
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import Month, PeriodRange


class MockExternalModel:
    """Deterministic ExternalModel-compatible mock for explain() integration tests.

    ``response_fn`` maps ``(location_name, n_future_periods) -> ndarray`` of
    ``n`` per-period predictions. The default produces a positive ramp so
    the ``log1p`` path stays clean and the surrogate sees variation.
    """

    name = "mock_model"

    def __init__(self, response_fn: Callable[[str, int], np.ndarray] | None = None):
        self.response_fn = response_fn or (lambda _loc, n: np.linspace(10.0, 50.0, n))

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        result: dict = {}
        for location in future_data.locations():
            fut_loc = future_data[location]
            n = len(fut_loc.time_period)
            preds = np.asarray(self.response_fn(location, n), dtype=float).reshape(-1, 1)
            result[location] = Samples(fut_loc.time_period, preds)  # type: ignore[call-arg]
        return DataSet(result, polygons=getattr(future_data, "polygons", None))


@pytest.fixture
def small_full_dataset() -> DataSet:
    """20-month, 2-location FullData fixture with enough variation for segmentation."""
    time_period = PeriodRange.from_time_periods(Month(2020, 1), Month(2021, 8))
    n = len(time_period)
    rng = np.random.default_rng(0)
    return DataSet(
        {
            "alpha": FullData(
                time_period,
                rng.normal(50, 5, n).round(1).clip(0).tolist(),
                rng.normal(25, 3, n).round(1).tolist(),
                rng.integers(10, 100, n).tolist(),
                [100_000] * n,
            ),
            "beta": FullData(
                time_period,
                rng.normal(80, 8, n).round(1).clip(0).tolist(),
                rng.normal(28, 4, n).round(1).tolist(),
                rng.integers(20, 150, n).tolist(),
                [200_000] * n,
            ),
        }
    )


def _explain_with_defaults(model, dataset, **overrides):
    """Run explain() with sane test defaults: small budget, no I/O side effects."""
    kwargs = {
        "model": model,
        "dataset": dataset,
        "location": "alpha",
        "horizon": 2,
        "num_perturbations": 20,
        "granularity": 4,
        "seed": 42,
        "plot": False,
        "save": False,
    }
    kwargs.update(overrides)
    return explain(**kwargs)


class TestExplainEndToEnd:
    def test_returns_sorted_list_of_named_coefficients(self, small_full_dataset):
        results = _explain_with_defaults(MockExternalModel(), small_full_dataset)

        assert isinstance(results, list)
        assert len(results) > 0
        for entry in results:
            assert isinstance(entry, tuple) and len(entry) == 2
            name, coef = entry
            assert isinstance(name, str) and name
            assert isinstance(coef, float)

    def test_results_sorted_by_absolute_coefficient_descending(self, small_full_dataset):
        results = _explain_with_defaults(MockExternalModel(), small_full_dataset)
        abs_coefs = [abs(c) for _, c in results]
        assert abs_coefs == sorted(abs_coefs, reverse=True)

    def test_feature_names_include_lags_and_future_steps(self, small_full_dataset):
        results = _explain_with_defaults(MockExternalModel(), small_full_dataset)
        names = {name for name, _ in results}
        # Historic features should be segmented into lags.
        assert any("_lag_" in name for name in names)
        # Future features should appear with _fut_N suffixes (horizon=2 -> _fut_1, _fut_2).
        assert any("_fut_" in name for name in names)

    def test_seeded_runs_are_deterministic(self, small_full_dataset):
        first = _explain_with_defaults(MockExternalModel(), small_full_dataset)
        second = _explain_with_defaults(MockExternalModel(), small_full_dataset)
        # Same seed, same model, same dataset -> identical explanation.
        assert first == second


class TestReturnMetrics:
    def test_return_metrics_true_yields_results_plus_metrics_dict(self, small_full_dataset):
        out = _explain_with_defaults(MockExternalModel(), small_full_dataset, return_metrics=True)
        assert isinstance(out, tuple) and len(out) == 2
        results, metrics = out
        assert isinstance(results, list)
        assert isinstance(metrics, dict)

    def test_metrics_contain_eloss_components(self, small_full_dataset):
        _, metrics = _explain_with_defaults(MockExternalModel(), small_full_dataset, return_metrics=True)
        # r2 + n_eff always populated; the eLoss extras are populated by the
        # return_metrics path.
        for key in ("r2", "n_eff", "delta_eloss", "auc_top_k", "auc_bottom_k"):
            assert key in metrics, f"missing metric: {key}"
            assert np.isfinite(metrics[key]), f"metric {key}={metrics[key]!r} is not finite"


class TestLog1pHelperEndToEnd:
    """The log1p clip+drop fix exercised through the real pipeline."""

    def test_uniformly_negative_predictions_warn_and_clip(self, small_full_dataset, caplog):
        model = MockExternalModel(response_fn=lambda _loc, n: np.full(n, -5.0))
        with caplog.at_level(logging.WARNING, logger="chap_core.explainability.lime"):
            results = _explain_with_defaults(model, small_full_dataset)

        # The pipeline still completes (previously crashed at `log1p`).
        assert isinstance(results, list) and len(results) > 0
        # And the visible diagnostic fires.
        assert any("clipping to 0" in rec.message and "negative" in rec.message for rec in caplog.records)

    def test_all_nan_predictions_raise_value_error(self, small_full_dataset):
        model = MockExternalModel(response_fn=lambda _loc, n: np.full(n, np.nan))
        with pytest.raises(ValueError, match="All perturbed predictions were non-finite"):
            _explain_with_defaults(model, small_full_dataset)

    def test_mixed_finite_and_non_finite_predictions_complete_without_length_mismatch(self, small_full_dataset, caplog):
        """Regression: r2_score raised on length mismatch when only some perturbations were non-finite.

        Before the fix, the surrogate fit ran on filtered (X_fit, z, weights) but the R²
        calculation called surrogate_model.predict(X) on the unfiltered X, so y_true and
        y_pred had different lengths and sklearn raised.
        """

        # Each perturbation is predicted on its own against the real location set,
        # so the explained location ("alpha") is queried once per perturbation.
        # Make every other such call return NaN to exercise the partial-drop path.
        alpha_calls = {"n": 0}

        def mixed_response(loc: str, n: int) -> np.ndarray:
            if loc == "alpha":
                alpha_calls["n"] += 1
                if alpha_calls["n"] % 2 == 0:
                    return np.full(n, np.nan)
            return np.linspace(10.0, 50.0, n)

        model = MockExternalModel(response_fn=mixed_response)
        with caplog.at_level(logging.WARNING, logger="chap_core.explainability.lime"):
            results = _explain_with_defaults(model, small_full_dataset)

        assert isinstance(results, list) and len(results) > 0
        assert any("non-finite" in rec.message and "dropping" in rec.message for rec in caplog.records), (
            "expected diagnostic about dropping non-finite perturbations"
        )


class TestSinglePredictionPath:
    """produce_lime_dataset predicts one perturbation at a time against the real
    location set — it does not batch perturbations under synthetic `pb_` ids."""

    def test_no_pseudo_locations_one_predict_per_perturbation(self, small_full_dataset):
        seen_locations: set[str] = set()
        predict_calls = {"n": 0}

        model = MockExternalModel()
        inner_predict = model.predict

        def recording_predict(historic_data, future_data):
            predict_calls["n"] += 1
            seen_locations.update(str(loc) for loc in future_data.locations())
            return inner_predict(historic_data, future_data)

        model.predict = recording_predict  # type: ignore[method-assign]

        n_perturbations = 12
        _explain_with_defaults(model, small_full_dataset, num_perturbations=n_perturbations)

        # No synthetic pseudo-location ever reaches the model...
        assert not any(loc.startswith("pb_") for loc in seen_locations), seen_locations
        # ...only the real dataset locations are used...
        assert seen_locations <= {"alpha", "beta"}, seen_locations
        # ...and the model is invoked once per perturbation.
        assert predict_calls["n"] == n_perturbations


class TestSaveAndPlot:
    """save=True should persist both the markdown report and the importance plot."""

    def test_save_writes_markdown_and_plot_png(self, small_full_dataset, tmp_path, monkeypatch):
        import matplotlib

        matplotlib.use("Agg")  # headless: no display needed
        from chap_core.explainability import lime as lime_module

        monkeypatch.setattr(lime_module, "CHAP_RUNS_DIR", tmp_path)

        _explain_with_defaults(MockExternalModel(), small_full_dataset, save=True, plot=True)

        explainability_dir = tmp_path / "explainability"
        md_files = list(explainability_dir.rglob("explanation.md"))
        png_files = list(explainability_dir.rglob("importance_plot.png"))

        assert len(md_files) == 1, "expected exactly one explanation.md"
        assert len(png_files) == 1, "expected the importance plot saved next to the markdown"
        assert png_files[0].parent == md_files[0].parent, "plot should sit beside the markdown"
        assert "importance_plot.png" in md_files[0].read_text(), "markdown should reference the plot"

    def test_two_runs_same_second_do_not_overwrite(self, tmp_path, monkeypatch):
        # Regression: multi-location runs share a model + can land in the same
        # second; the per-run dir must stay unique so explanation.md isn't clobbered.
        from chap_core.explainability import lime as lime_module
        from chap_core.explainability.lime import save_explanation

        monkeypatch.setattr(lime_module, "CHAP_RUNS_DIR", tmp_path)

        args = dict(
            results=[("rainfall_lag_0", 0.5)],
            model_name="mock_model",
            location="alpha",
            horizon=1,
            r2=0.5,
            n_eff=3.0,
            params={},
        )
        first = save_explanation(**args)
        second = save_explanation(**args)

        assert first != second, "second run must not reuse the first run's path"
        assert first.exists() and second.exists()
        assert len(list((tmp_path / "explainability").rglob("explanation.md"))) == 2

    def test_no_save_writes_nothing(self, small_full_dataset, tmp_path, monkeypatch):
        from chap_core.explainability import lime as lime_module

        monkeypatch.setattr(lime_module, "CHAP_RUNS_DIR", tmp_path)

        _explain_with_defaults(MockExternalModel(), small_full_dataset, save=False, plot=False)

        assert not (tmp_path / "explainability").exists(), "save=False must not write anything"


class TestExplainAdaptiveEndToEnd:
    """Smoke test for the adaptive variant — same contract, different sampling strategy."""

    def test_adaptive_returns_sorted_coefficients(self, small_full_dataset):
        results = explain_adaptive(
            model=MockExternalModel(),
            dataset=small_full_dataset,
            location="alpha",
            horizon=2,
            num_perturbations=20,
            granularity=4,
            seed=42,
            plot=False,
            save=False,
        )
        assert isinstance(results, list) and len(results) > 0
        abs_coefs = [abs(c) for _, c in results]
        assert abs_coefs == sorted(abs_coefs, reverse=True)

    def test_adaptive_return_metrics_yields_eloss_components(self, small_full_dataset):
        # The metrics end-to-end path is covered for explain() in TestReturnMetrics;
        # explain_adaptive populates the same eLoss components and needs its own.
        out = explain_adaptive(
            model=MockExternalModel(),
            dataset=small_full_dataset,
            location="alpha",
            horizon=2,
            num_perturbations=20,
            granularity=4,
            seed=42,
            plot=False,
            save=False,
            return_metrics=True,
        )
        assert isinstance(out, tuple) and len(out) == 2
        results, metrics = out
        assert isinstance(results, list)
        for key in ("r2", "n_eff", "delta_eloss", "auc_top_k", "auc_bottom_k"):
            assert key in metrics, f"missing metric: {key}"
            assert np.isfinite(metrics[key]), f"metric {key}={metrics[key]!r} is not finite"
