"""Prediction-time logic for the stacking ensemble."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from chap_core.datatypes import Samples
from chap_core.ensemble._sample_extractor import SampleExtractor as _SampleExtractor
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

if TYPE_CHECKING:
    import pandas as pd

    from chap_core.ensemble._meta_models import NonNegativeMetaModel, ProbabilisticMetaModel


class EnsemblePredictor:
    def __init__(
        self,
        predictors: list[Any],
        meta: NonNegativeMetaModel | ProbabilisticMetaModel,
        probabilistic: bool,
        n_samples: int,
        use_residual_bootstrap: bool = False,
        base_residuals: list[np.ndarray] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._predictors = list(predictors)
        self._meta = meta
        self._prob = probabilistic
        self._n_samples = n_samples
        self._use_residual_bootstrap = use_residual_bootstrap
        self._base_residuals = base_residuals or []
        self._rng = rng or np.random.default_rng()

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet[Samples]:
        df_future = future_data.to_pandas()
        key_cols = ["location", "time_period"]

        if self._prob:
            base_samp = [
                _SampleExtractor.reshape_samples(
                    p.predict(historic_data, future_data), df_future, self._n_samples, rng=self._rng
                )
                for p in self._predictors
            ]
            ens_samp = self._meta.predict(base_samp)  # type: ignore[arg-type]
            return self._pack_samples(ens_samp, df_future, future_data)

        meta_cols = []
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
            s_count = self._n_samples
            ens_samp = np.zeros((n_rows, s_count), float)
            for model_idx, residuals in enumerate(self._base_residuals):
                res_clean = residuals[~np.isnan(residuals)]
                if len(res_clean) == 0:
                    for row_idx in range(n_rows):
                        ens_samp[row_idx, :] += w[model_idx] * X_meta_future[row_idx, model_idx]
                    continue
                for row_idx in range(n_rows):
                    sampled_res = self._rng.choice(res_clean, size=s_count, replace=True)
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
            result[loc] = Samples(tp, preds_loc.reshape(-1, 1).astype(float))  # type: ignore[call-arg]
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
            result[loc] = Samples(tp, all_samples[loc_idx, :])  # type: ignore[call-arg]
        return DataSet(result)


__all__ = ["EnsemblePredictor"]
