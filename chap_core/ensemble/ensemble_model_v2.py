"""Minimal stacking ensemble for CHAP (v2)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

from chap_core.datatypes import FullData, Samples
from chap_core.models.configured_model import ConfiguredModel
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

if TYPE_CHECKING:
    from chap_core.models.model_template import ModelTemplate

logger = logging.getLogger(__name__)


def _crps_score(obs: np.ndarray, forecast: np.ndarray) -> float:
    term1 = np.mean(np.abs(forecast - obs.reshape(-1, 1)), axis=1)
    m = forecast.shape[1]
    if m <= 1:
        return float(np.mean(term1))
    term2 = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            term2 += np.mean(np.abs(forecast[:, i] - forecast[:, j]))
    term2 /= m * (m - 1) / 2.0
    return float(np.mean(term1) - 0.5 * term2)


class NonNegativeMetaModel:
    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NonNegativeMetaModel:
        coef, _ = nnls(X, y)  # type: ignore[misc]
        s = coef.sum()
        self.coef_ = coef / s if s > 0 else coef
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Meta-model not fitted")
        return np.dot(X, self.coef_)


class ProbabilisticMetaModel:
    def __init__(self, verbose: bool = False) -> None:
        self.coef_: np.ndarray | None = None
        self.verbose = verbose

    def fit(self, X_samples: list[np.ndarray], y: np.ndarray) -> ProbabilisticMetaModel:
        target_shape = X_samples[0].shape
        for i, s in enumerate(X_samples):
            if s.shape != target_shape:
                raise ValueError(f"Shape mismatch: X_samples[0]={target_shape}, X_samples[{i}]={s.shape}")

        def obj(w: np.ndarray) -> float:
            w_norm = w / (w.sum() + 1e-10)
            ens = sum(w_norm[i] * X_samples[i] for i in range(len(X_samples)))
            return _crps_score(y, ens)

        n = len(X_samples)
        w0 = np.ones(n) / n
        res = minimize(
            obj,
            w0,
            method="SLSQP",
            constraints={"type": "ineq", "fun": lambda w: w},
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        if self.verbose:
            logger.info("Probabilistic meta-model fit: CRPS=%.4f, success=%s", res.fun, res.success)
        coef = res.x
        coef = coef / (coef.sum() + 1e-10)
        self.coef_ = coef
        return self

    def predict(self, X_samples: list[np.ndarray]) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Meta-model not fitted")
        ens = sum(self.coef_[i] * X_samples[i] for i in range(len(X_samples)))
        return np.maximum(ens, 0.0)


class _SampleExtractor:
    @staticmethod
    def samples_to_flat(preds_ds: Samples) -> pd.DataFrame:
        df = preds_ds.to_pandas()
        if "forecast" in df.columns:
            pred_col = "forecast"
        elif "value" in df.columns:
            pred_col = "value"
        else:
            sample_cols = [c for c in df.columns if c.startswith("sample_")]
            if sample_cols:
                df["forecast"] = df[sample_cols].mean(axis=1)
                pred_col = "forecast"
            else:
                raise ValueError(f"No forecast/value/sample_* in columns: {list(df.columns)}")
        if "horizon_distance" in df.columns:
            df = df[df["horizon_distance"] == 0].copy()
        missing = [c for c in ("location", "time_period") if c not in df.columns]
        if missing:
            raise ValueError(f"Missing {missing} in prediction DataFrame")
        out = df[["location", "time_period", pred_col]].copy()
        return out.rename(columns={pred_col: "forecast"})

    @staticmethod
    def reshape_samples(preds_ds: Samples, df_ref: pd.DataFrame, target_n: int) -> np.ndarray:
        df_pred = preds_ds.to_pandas()
        sample_cols = [c for c in df_pred.columns if c.startswith("sample_")]
        if sample_cols:
            mat = df_pred[sample_cols].to_numpy(float)
        else:
            df_flat = _SampleExtractor.samples_to_flat(preds_ds)
            merged = df_ref[["location", "time_period"]].merge(df_flat, on=["location", "time_period"], how="left")
            pts = merged["forecast"].to_numpy()
            mat = np.tile(pts.reshape(-1, 1), (1, target_n))
            return mat
        n_rows, n_samp = mat.shape
        if n_samp != target_n:
            if n_samp == 1:
                mat = np.tile(mat, (1, target_n))
            else:
                idx = np.random.choice(n_samp, target_n, replace=True)
                mat = mat[:, idx]
        assert mat.shape == (n_rows, target_n)
        return mat


class EnsembleModel(ConfiguredModel):
    def __init__(
        self,
        base_templates: Sequence[ModelTemplate] | None = None,
        method: str = "probabilistic",
        inner_val_periods: int = 12,
        target_col: str = "disease_cases",
        n_samples: int = 100,
        use_residual_bootstrap: bool = False,
    ) -> None:
        super().__init__()
        self.base_templates = list(base_templates or [])
        if not self.base_templates:
            raise ValueError("Need at least one base model")
        if method not in ("deterministic", "probabilistic"):
            raise ValueError(method)
        self.method = method
        self.inner_val_periods = inner_val_periods
        self.target_col = target_col
        self.n_samples = n_samples
        self.use_residual_bootstrap = use_residual_bootstrap
        self.meta_model: NonNegativeMetaModel | ProbabilisticMetaModel | None = None
        self.weights: np.ndarray | None = None
        self._base_residuals: list[np.ndarray] = []

    def _base_names(self) -> list[str]:
        names: list[str] = []
        for tmpl in self.base_templates:
            name = getattr(tmpl, "name", None)
            if not name:
                repo = getattr(tmpl, "repo", None)
                if isinstance(repo, str) and repo:
                    name = repo.rstrip("/").split("/")[-1]
                else:
                    name = str(tmpl)
            names.append(name)
        return names

    def train(self, train_data: DataSet, extra_args: Any = None) -> EnsemblePredictor:
        df = train_data.to_pandas()
        all_periods = sorted(df["time_period"].dropna().astype(str).unique())
        if len(all_periods) <= self.inner_val_periods:
            split_idx = len(all_periods) // 2
        else:
            split_idx = len(all_periods) - self.inner_val_periods
        logger.info(
            "Inner split: %d periods, train=%d, val=%d", len(all_periods), split_idx, len(all_periods) - split_idx
        )

        train_mask = df["time_period"].astype(str).isin(set(all_periods[:split_idx]))
        inner_train = DataSet.from_pandas(df[train_mask], FullData, fill_missing=True)
        val_data = DataSet.from_pandas(df[~train_mask], FullData, fill_missing=True)

        ests = [t.get_model(None)() for t in self.base_templates]  # type: ignore[call-arg]
        preds_inner = [e.train(inner_train) for e in ests]

        df_val = val_data.to_pandas()
        y_val = df_val[self.target_col].to_numpy()
        key_cols = ["location", "time_period"]

        self._base_residuals = []
        if self.use_residual_bootstrap:
            for p in preds_inner:
                preds_ds = p.predict(inner_train, val_data)
                df_pred = _SampleExtractor.samples_to_flat(preds_ds)
                merged = df_val[key_cols].merge(df_pred, on=key_cols, how="left")
                res = y_val - merged["forecast"].to_numpy()
                self._base_residuals.append(res[~np.isnan(res)])

        if self.method == "probabilistic":
            meta_list = [
                _SampleExtractor.reshape_samples(p.predict(inner_train, val_data), df_val, self.n_samples)
                for p in preds_inner
            ]
        else:
            cols = []
            for p in preds_inner:
                preds_ds = p.predict(inner_train, val_data)
                df_pred = _SampleExtractor.samples_to_flat(preds_ds)
                merged = df_val[key_cols].merge(df_pred, on=key_cols, how="left")
                cols.append(merged["forecast"].to_numpy())
            meta_mat = np.column_stack(cols)

        mask = ~np.isnan(y_val)
        if not np.any(mask):
            raise ValueError("No valid targets in validation")
        y_clean = y_val[mask]
        if self.method == "probabilistic":
            X_clean = [m[mask, :] for m in meta_list]
            self.meta_model = ProbabilisticMetaModel(verbose=True).fit(X_clean, y_clean)
        else:
            X_clean = meta_mat[mask, :]
            self.meta_model = NonNegativeMetaModel().fit(X_clean, y_clean)

        coef = np.maximum(np.asarray(self.meta_model.coef_, float), 0.0)  # type: ignore[arg-type]
        s = coef.sum()
        if s > 0:
            w_pct = coef / s * 100.0
        else:
            n = len(coef)
            w_pct = np.full(n, 100.0 / n)
        self.weights = w_pct

        names = self._base_names()
        logger.info("Meta-weights (percent): %s", self.weights)
        for name, w in zip(names, self.weights, strict=True):
            logger.info("  %s: %.2f%%", name, w)
        try:
            report_path = Path("ensemble_meta_report_v2.csv")
            header = "Model," + ",".join(names) + "\n"
            row = "ensemble_meta," + ",".join(f"{float(w):.6f}" for w in self.weights) + "\n"
            report_path.write_text(header + row, encoding="utf-8")
            logger.info("Saved ensemble meta report to %s", report_path.resolve())
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to write ensemble_meta_report_v2.csv: %s", e)

        full_ests = [t.get_model(None)() for t in self.base_templates]  # type: ignore[call-arg]
        full_predictors = [e.train(train_data) for e in full_ests]

        return EnsemblePredictor(
            target_col=self.target_col,
            predictors=full_predictors,
            meta=self.meta_model,
            probabilistic=(self.method == "probabilistic"),
            n_samples=self.n_samples,
            use_residual_bootstrap=self.use_residual_bootstrap,
            base_residuals=self._base_residuals,
        )

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        raise NotImplementedError("Use train() to obtain EnsemblePredictor")


class EnsemblePredictor:
    def __init__(
        self,
        target_col: str,
        predictors: Sequence[Any],
        meta: NonNegativeMetaModel | ProbabilisticMetaModel,
        probabilistic: bool,
        n_samples: int,
        use_residual_bootstrap: bool = False,
        base_residuals: list[np.ndarray] | None = None,
    ) -> None:
        self._target_col = target_col
        self._predictors = list(predictors)
        self._meta = meta
        self._prob = probabilistic
        self._n_samples = n_samples
        self._use_residual_bootstrap = use_residual_bootstrap
        self._base_residuals = base_residuals or []

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet[Samples]:
        df_future = future_data.to_pandas()
        key_cols = ["location", "time_period"]

        if self._prob:
            base_samp = [
                _SampleExtractor.reshape_samples(p.predict(historic_data, future_data), df_future, self._n_samples)
                for p in self._predictors
            ]
            ens_samp = self._meta.predict(base_samp)  # type: ignore[arg-type]
            return self._pack_samples(ens_samp, df_future, future_data)

        meta_cols: list[np.ndarray] = []
        for p in self._predictors:
            preds_ds = p.predict(historic_data, future_data)
            df_pred = _SampleExtractor.samples_to_flat(preds_ds)
            merged = df_future[key_cols].merge(df_pred, on=key_cols, how="left")
            meta_cols.append(merged["forecast"].to_numpy())
        X_meta_future = np.column_stack(meta_cols)
        y_point = self._meta.predict(X_meta_future)  # type: ignore[arg-type]

        if self._use_residual_bootstrap and self._base_residuals:
            w = np.asarray(self._meta.coef_, float)  # type: ignore[arg-type]
            w = np.maximum(w, 0.0)
            s = w.sum()
            if s <= 0:
                raise ValueError("Meta weights sum <= 0")
            w /= s
            n_rows = X_meta_future.shape[0]
            S = self._n_samples
            ens_samp = np.zeros((n_rows, S), float)
            for model_idx, residuals in enumerate(self._base_residuals):
                res_clean = residuals[~np.isnan(residuals)]
                if len(res_clean) == 0:
                    for row_idx in range(n_rows):
                        base_pred = X_meta_future[row_idx, model_idx]
                        ens_samp[row_idx, :] += w[model_idx] * base_pred
                    continue
                for row_idx in range(n_rows):
                    sampled_res = np.random.choice(res_clean, size=S, replace=True)
                    base_pred = X_meta_future[row_idx, model_idx]
                    model_samples = np.maximum(base_pred + sampled_res, 0.0)
                    ens_samp[row_idx, :] += w[model_idx] * model_samples
            return self._pack_samples(ens_samp, df_future, future_data)

        df_out = df_future.copy()
        df_out["forecast"] = y_point
        df_out = df_out.sort_values(key_cols)
        result: dict[Any, Samples] = {}
        for loc in sorted(df_out["location"].unique()):
            mask = df_out["location"] == loc
            df_loc = df_out[mask].copy()
            tp = future_data[loc].time_period
            preds_loc = df_loc["forecast"].to_numpy()
            if len(preds_loc) != len(tp):
                raise ValueError(f"Length mismatch for location {loc!r}")
            samples_arr = preds_loc.reshape(-1, 1).astype(float)
            result[loc] = Samples(samples=samples_arr, time_period=tp)
        return DataSet(result)

    @staticmethod
    def _pack_samples(all_samples: np.ndarray, df_future: pd.DataFrame, future_data: DataSet) -> DataSet[Samples]:
        result: dict[Any, Samples] = {}
        for loc in sorted(future_data.locations()):
            mask = (df_future["location"] == loc).to_numpy()
            loc_idx = np.where(mask)[0]
            tp = future_data[loc].time_period
            if len(loc_idx) != len(tp):
                raise ValueError(f"Row/time_period mismatch for {loc}")
            loc_samples = all_samples[loc_idx, :]
            result[loc] = Samples(samples=loc_samples, time_period=tp)
        return DataSet(result)
