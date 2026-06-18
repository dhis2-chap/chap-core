"""Prediction-time logic for the stacking ensemble."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from chap_core.datatypes import Samples
from chap_core.ensemble._sample_extractor import SampleExtractor as _SampleExtractor
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

if TYPE_CHECKING:
    from chap_core.ensemble._meta_models import NonNegativeMetaModel, ProbabilisticMetaModel

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    def __init__(
        self,
        predictors: list[Any],
        meta: NonNegativeMetaModel | ProbabilisticMetaModel,
        probabilistic: bool,
        n_samples: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._predictors = list(predictors)
        self._meta = meta
        self._prob = probabilistic
        self._n_samples = n_samples
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
            meta_prob = cast("ProbabilisticMetaModel", self._meta)
            ens_samp = meta_prob.predict(base_samp)
            return self._pack_samples(ens_samp, df_future, future_data)

        meta_cols = []
        for p in self._predictors:
            preds_ds = p.predict(historic_data, future_data)
            df_pred = _SampleExtractor.samples_to_flat(preds_ds)
            merged = df_future[key_cols].merge(df_pred, on=key_cols, how="left")
            meta_cols.append(merged["forecast"].to_numpy())
        X_meta_future = np.column_stack(meta_cols)
        if np.any(np.isnan(X_meta_future)):
            raise ValueError("Missing base model predictions for one or more rows")
        meta_det = cast("NonNegativeMetaModel", self._meta)
        y_point = meta_det.predict(X_meta_future)

        logger.warning(
            "Deterministic ensemble outputs a single sample; CRPS reduces to MAE and is not comparable to multi-sample CRPS"
        )
        df_out = df_future.copy()
        df_out["forecast"] = y_point
        df_out = df_out.sort_values(key_cols)
        result: dict[Any, Samples] = {}
        for loc in sorted(df_out["location"].unique()):
            mask = df_out["location"] == loc
            df_loc = df_out[mask].copy()
            tp = future_data[loc].time_period
            df_loc["time_period"] = df_loc["time_period"].astype(str)
            tp_values = [str(tp_val) for tp_val in tp.topandas()]
            df_loc = df_loc.set_index("time_period").reindex(tp_values).reset_index()
            if df_loc["forecast"].isna().any():
                raise ValueError(f"Missing deterministic forecasts for location {loc!r}")
            preds_loc = df_loc["forecast"].to_numpy()
            if len(preds_loc) != len(tp_values):
                raise ValueError(f"Length mismatch for location {loc!r}")
            df_samples = pd.DataFrame({"time_period": tp.topandas()})
            df_samples["sample_0"] = preds_loc
            result[loc] = Samples.from_pandas(df_samples)
        return DataSet(result)

    @staticmethod
    def _pack_samples(all_samples: np.ndarray, df_future: pd.DataFrame, future_data: DataSet) -> DataSet[Samples]:
        result: dict[Any, Samples] = {}
        n_samples = all_samples.shape[1]
        sample_cols = [f"sample_{i}" for i in range(n_samples)]
        for loc in sorted(future_data.locations()):
            mask = (df_future["location"] == loc).to_numpy()
            loc_idx = np.where(mask)[0]
            tp = future_data[loc].time_period
            if len(loc_idx) != len(tp):
                raise ValueError(f"Row/time_period mismatch for {loc}")
            df_samples = pd.DataFrame({"time_period": tp.topandas()})
            df_samples[sample_cols] = all_samples[loc_idx, :]
            result[loc] = Samples.from_pandas(df_samples)
        return DataSet(result)


__all__ = ["EnsemblePredictor"]
