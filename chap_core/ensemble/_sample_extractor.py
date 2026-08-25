"""Helpers for flattening and reshaping sample-based forecasts."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from chap_core.datatypes import Samples

logger = logging.getLogger(__name__)


def _resample_to_quantiles(mat: np.ndarray, target_n: int) -> np.ndarray:
    """Resample each row to ``target_n`` samples via its empirical quantile function.

    Downstream vincentization sorts the samples and treats them as quantile
    estimates, so interpolating the per-row quantile function preserves the
    spread exactly. Drawing indices at random would instead reuse the same
    draws for every row and distort every location's spread identically.
    """
    n_rows, n_samp = mat.shape
    sorted_mat = np.sort(mat, axis=1)
    src = (np.arange(n_samp) + 0.5) / n_samp
    dst = (np.arange(target_n) + 0.5) / target_n
    out = np.empty((n_rows, target_n), dtype=float)
    for i in range(n_rows):
        out[i] = np.interp(dst, src, sorted_mat[i])
    return out


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
                logger.warning(
                    "Collapsing probabilistic samples to a point forecast using the median; uncertainty is discarded"
                )
                df["forecast"] = df[sample_cols].median(axis=1)
                pred_col = "forecast"
            else:
                raise ValueError(f"No forecast/value/sample_* in columns: {list(df.columns)}")
        missing = [c for c in ("location", "time_period") if c not in df.columns]
        if missing:
            raise ValueError(f"Missing {missing} in prediction DataFrame")
        out = df[["location", "time_period", pred_col]].copy()
        return out.rename(columns={pred_col: "forecast"})

    @staticmethod
    def reshape_samples(preds_ds: Samples, df_ref: pd.DataFrame, target_n: int) -> np.ndarray:
        df_pred = pd.DataFrame(preds_ds.to_pandas())

        # Always align on location/time_period first.
        key_cols = ["location", "time_period"]
        if not all(c in df_pred.columns for c in key_cols):
            # Fall back to row order; this is less robust.
            sample_cols = [c for c in df_pred.columns if c.startswith("sample_")]
            if sample_cols:
                if len(df_pred) != len(df_ref):
                    raise ValueError(
                        f"Cannot align predictions by row order: got {len(df_pred)} prediction rows "
                        f"for {len(df_ref)} reference rows. Predictions are missing "
                        f"{', '.join(key_cols)} columns needed for a reliable merge."
                    )
                logger.warning(
                    "Predictions lack %s; falling back to row-order alignment, which assumes the base model "
                    "returns rows in the same order as the reference frame",
                    ", ".join(key_cols),
                )
                mat = df_pred[sample_cols].to_numpy(float)
            else:
                df_flat = SampleExtractor.samples_to_flat(preds_ds)
                merged = df_ref[key_cols].merge(df_flat, on=key_cols, how="left")
                pts = merged["forecast"].to_numpy()
                logger.warning(
                    "Probabilistic predictions missing samples; repeating point forecasts for %d samples", target_n
                )
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
                logger.warning(
                    "Probabilistic predictions missing samples; repeating point forecasts for %d samples", target_n
                )
                return np.tile(pts.reshape(-1, 1), (1, target_n))

        _, n_samp = mat.shape
        if n_samp != target_n:
            if n_samp == 1:
                mat = np.tile(mat, (1, target_n))
            else:
                mat = _resample_to_quantiles(mat, target_n)
        return mat
