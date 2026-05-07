"""Helpers for flattening and reshaping sample-based forecasts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from chap_core.datatypes import Samples


class SampleExtractor:
    @staticmethod
    def samples_to_flat(preds_ds: Samples) -> pd.DataFrame:
        df = preds_ds.to_pandas()
        df = pd.DataFrame(df)
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
    def reshape_samples(
        preds_ds: Samples, df_ref: pd.DataFrame, target_n: int, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        rng = rng or np.random.default_rng()
        df_pred = pd.DataFrame(preds_ds.to_pandas())

        # Always align on location/time_period first.
        key_cols = ["location", "time_period"]
        if not all(c in df_pred.columns for c in key_cols):
            # Fall back to row order; this is less robust.
            sample_cols = [c for c in df_pred.columns if c.startswith("sample_")]
            if sample_cols:
                mat = df_pred[sample_cols].to_numpy(float)
            else:
                df_flat = SampleExtractor.samples_to_flat(preds_ds)
                merged = df_ref[key_cols].merge(df_flat, on=key_cols, how="left")
                pts = merged["forecast"].to_numpy()
                return np.tile(pts.reshape(-1, 1), (1, target_n))
        else:
            # Align via merge.
            sample_cols = [c for c in df_pred.columns if c.startswith("sample_")]
            if sample_cols:
                merged = df_ref[key_cols].merge(df_pred[key_cols + sample_cols], on=key_cols, how="left")
                mat = merged[sample_cols].to_numpy(float)
            else:
                df_flat = SampleExtractor.samples_to_flat(preds_ds)
                merged = df_ref[key_cols].merge(df_flat, on=key_cols, how="left")
                pts = merged["forecast"].to_numpy()
                return np.tile(pts.reshape(-1, 1), (1, target_n))

        _, n_samp = mat.shape
        if n_samp != target_n:
            if n_samp == 1:
                mat = np.tile(mat, (1, target_n))
            else:
                idx = rng.choice(n_samp, target_n, replace=True)
                mat = mat[:, idx]
        return mat
