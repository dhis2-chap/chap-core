"""Minimal, robust stacking ensemble for CHAP."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from chap_core.ensemble._meta_models import (
    NonNegativeMetaModel,
    ProbabilisticMetaModel,
    crps_ensemble,
)
from chap_core.ensemble._predictor import EnsemblePredictor
from chap_core.ensemble._sample_extractor import SampleExtractor as _SampleExtractor
from chap_core.ensemble.wrappers import BaseModelSpec, TemplateWithConfig
from chap_core.models.configured_model import ConfiguredModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from chap_core.database.model_templates_and_config_tables import ModelTemplateInformation
    from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

logger = logging.getLogger(__name__)


class EnsembleModel(ConfiguredModel):
    @property
    def model_information(self) -> ModelTemplateInformation | None:
        return None

    def __init__(
        self,
        base_templates: Sequence[Any] | None = None,
        method: str = "probabilistic",
        inner_val_periods: int = 12,
        horizon: int = 3,
        target_col: str = "disease_cases",
        n_samples: int = 100,
        meta_model: NonNegativeMetaModel | ProbabilisticMetaModel | None = None,
    ) -> None:
        super().__init__()
        self.base_templates = list(base_templates or [])
        if not self.base_templates:
            raise ValueError("Need at least one base model")
        if method not in ("deterministic", "probabilistic"):
            raise ValueError(method)
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")
        self.method = method
        self.inner_val_periods = inner_val_periods
        self.horizon = horizon
        self.target_col = target_col
        self.n_samples = n_samples
        self.meta_model: NonNegativeMetaModel | ProbabilisticMetaModel | None = meta_model
        self.weights: np.ndarray | None = None

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

    def inner_validation_windows(self, train_data: DataSet) -> list[tuple[DataSet, DataSet]]:
        """Split the tail of ``train_data`` into (historic, future) windows of ``horizon`` periods.

        Base models are asked for exactly ``horizon`` steps in each window, matching the
        horizon they are used at during the outer backtest. Fitting the meta-weights on a
        single long window instead would rank the base models at the wrong horizon, since
        relative forecast skill is strongly horizon-dependent.
        """
        periods = list(train_data.period_range)
        if len(periods) < 2:
            raise ValueError("Need at least two time periods for training")
        split_idx = (
            len(periods) // 2 if len(periods) <= self.inner_val_periods else len(periods) - self.inner_val_periods
        )
        if split_idx <= 0 or split_idx >= len(periods):
            raise ValueError("Invalid inner validation split")

        windows: list[tuple[DataSet, DataSet]] = []
        for start in range(split_idx, len(periods), self.horizon):
            stop = min(start + self.horizon, len(periods))
            historic = train_data.restrict_time_period(slice(None, periods[start - 1]))
            future = train_data.restrict_time_period(slice(periods[start], periods[stop - 1]))
            windows.append((historic, future))

        logger.info(
            "Inner validation: %d periods, train=%d, val=%d, %d window(s) of horizon %d",
            len(periods),
            split_idx,
            len(periods) - split_idx,
            len(windows),
            self.horizon,
        )
        return windows

    def train(self, train_data: DataSet, extra_args: Any = None) -> EnsemblePredictor:
        windows = self.inner_validation_windows(train_data)
        inner_train = windows[0][0]

        ests: list[Any] = []
        for tmpl in self.base_templates:
            est_cls = cast("type[Any]", tmpl.get_model(None))
            ests.append(est_cls())
        preds_inner = [e.train(inner_train) for e in ests]

        key_cols = ["location", "time_period"]
        df_val = pd.concat([w[1].to_pandas() for w in windows], ignore_index=True)
        y_val = df_val[self.target_col].to_numpy()

        # The target must never reach the base models: ExternalModel writes future_data
        # verbatim to the CSV it hands the model, so leaving disease_cases in place would
        # let a base model read the very values the meta-weights are fitted against.
        masked_windows = [(historic, future.remove_field(self.target_col)) for historic, future in windows]

        meta_list: list[np.ndarray] | None = None
        meta_mat: np.ndarray | None = None
        if self.method == "probabilistic":
            meta_list = []
            for p in preds_inner:
                per_window = [
                    _SampleExtractor.reshape_samples(
                        p.predict(historic, future),
                        future.to_pandas(),
                        self.n_samples,
                    )
                    for historic, future in masked_windows
                ]
                meta_list.append(np.concatenate(per_window, axis=0))
        else:
            cols = []
            for p in preds_inner:
                per_window = []
                for historic, future in masked_windows:
                    preds_ds = p.predict(historic, future)
                    df_pred = _SampleExtractor.samples_to_flat(preds_ds)
                    merged = future.to_pandas()[key_cols].merge(df_pred, on=key_cols, how="left")
                    per_window.append(merged["forecast"].to_numpy())
                cols.append(np.concatenate(per_window))
            meta_mat = np.column_stack(cols)

        nan_in_features = np.zeros(len(y_val), dtype=bool)
        if self.method == "probabilistic":
            assert meta_list is not None
            per_base_nan = []
            for arr in meta_list:
                nan_rows = np.any(np.isnan(arr), axis=1)
                nan_in_features |= nan_rows
                per_base_nan.append(int(np.sum(nan_rows)))
        else:
            assert meta_mat is not None
            nan_in_features = np.any(np.isnan(meta_mat), axis=1)
            per_base_nan = [int(np.sum(np.isnan(meta_mat[:, i]))) for i in range(meta_mat.shape[1])]

        dropped = int(np.sum(nan_in_features | np.isnan(y_val)))
        if dropped:
            logger.warning("Dropping %d validation rows due to NaNs in targets/features", dropped)
            names = self._base_names()
            for name, cnt in zip(names, per_base_nan, strict=False):
                if cnt:
                    logger.warning("NaN count for base model %s: %d", name, cnt)

        mask = ~np.isnan(y_val) & ~nan_in_features
        if not np.any(mask):
            raise ValueError("No valid targets in validation")
        y_clean = y_val[mask]
        if self.method == "probabilistic":
            assert meta_list is not None
            X_clean_samples = [m[mask, :] for m in meta_list]
            if self.meta_model is None:
                self.meta_model = ProbabilisticMetaModel(verbose=True)
            meta_model_prob = cast("ProbabilisticMetaModel", self.meta_model)
            meta_model_prob.fit(X_clean_samples, y_clean)
        else:
            assert meta_mat is not None
            X_clean_mat = meta_mat[mask, :]
            if self.meta_model is None:
                self.meta_model = NonNegativeMetaModel()
            meta_model_det = cast("NonNegativeMetaModel", self.meta_model)
            meta_model_det.fit(X_clean_mat, y_clean)

        assert self.meta_model is not None
        coef_raw = cast("np.ndarray", self.meta_model.coef_)
        coef = np.maximum(np.asarray(coef_raw, float), 0.0)
        total = float(np.sum(coef))
        if total <= 0:
            # Both meta-models fall back to uniform weights rather than returning an
            # all-zero solution, so this should be unreachable.
            raise ValueError("Meta-model produced non-positive weights")
        self.weights = coef / total * 100.0

        names = self._base_names()
        assert self.weights is not None
        logger.info("Meta-weights (percent): %s", self.weights)
        for name, w in zip(names, self.weights, strict=True):
            logger.info("  %s: %.2f%%", name, w)

        full_ests: list[Any] = []
        for tmpl in self.base_templates:
            est_cls = cast("type[Any]", tmpl.get_model(None))
            full_ests.append(est_cls())
        full_predictors = [e.train(train_data) for e in full_ests]

        return EnsemblePredictor(
            predictors=full_predictors,
            meta=self.meta_model,
            probabilistic=(self.method == "probabilistic"),
            n_samples=self.n_samples,
        )

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        raise NotImplementedError("Use train() to obtain EnsemblePredictor")


class EnsembleEstimator(EnsembleModel):
    """Legacy class name/API backed by the same core implementation."""

    def __init__(
        self,
        base_model_templates: list[Any] | None = None,
        base_model_specs: Sequence[BaseModelSpec] | None = None,
        target_column: str = "disease_cases",
        inner_val_periods: int = 12,
        horizon: int = 3,
        meta_model: Any | None = None,
        probabilistic_meta_model: bool = False,
        n_samples: int = 100,
        **kwargs: Any,
    ) -> None:
        del kwargs
        specs = list(base_model_specs or [])
        if base_model_templates is not None:
            specs.extend(BaseModelSpec(template=t, config=None) for t in base_model_templates)
        if not specs:
            raise ValueError("EnsembleEstimator requires at least one base model.")

        self._base_specs = specs
        method = "probabilistic" if probabilistic_meta_model else "deterministic"
        super().__init__(
            base_templates=[TemplateWithConfig(s.template, s.config) for s in specs],
            method=method,
            inner_val_periods=inner_val_periods,
            horizon=horizon,
            target_col=target_column,
            n_samples=n_samples,
            meta_model=meta_model,
        )

    @classmethod
    def from_config(cls, spec: Any) -> EnsembleEstimator:
        base_specs = [
            BaseModelSpec(template=bm["template"], config=bm.get("config")) for bm in spec.config["base_models"]
        ]
        return cls(
            base_model_specs=base_specs,
            target_column=spec.config.get("target_column", "disease_cases"),
            inner_val_periods=spec.config.get("inner_val_periods", 12),
        )

    def train(self, train_data: DataSet, extra_args: Any = None) -> EnsemblePredictor:
        pred = super().train(train_data, extra_args)
        return pred


__all__ = [
    "BaseModelSpec",
    "EnsembleEstimator",
    "EnsembleModel",
    "NonNegativeMetaModel",
    "ProbabilisticMetaModel",
    "crps_ensemble",
]
