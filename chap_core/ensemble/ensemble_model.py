"""Minimal, robust stacking ensemble for CHAP."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from chap_core.assessment.dataset_splitting import train_test_split
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
        target_col: str = "disease_cases",
        n_samples: int = 100,
        meta_model: NonNegativeMetaModel | ProbabilisticMetaModel | None = None,
        random_state: int | None = None,
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
        self.meta_model: NonNegativeMetaModel | ProbabilisticMetaModel | None = meta_model
        self.weights: np.ndarray | None = None
        self.random_state: int | None = random_state

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
        # Local RNG for reproducibility.
        rng = np.random.default_rng(self.random_state)

        periods = list(train_data.period_range)
        if len(periods) < 2:
            raise ValueError("Need at least two time periods for training")
        split_idx = (
            len(periods) // 2 if len(periods) <= self.inner_val_periods else len(periods) - self.inner_val_periods
        )
        if split_idx <= 0 or split_idx >= len(periods):
            raise ValueError("Invalid inner validation split")
        split_period = periods[split_idx]

        logger.info(
            "Inner split: %d periods, train=%d, val=%d",
            len(periods),
            split_idx,
            len(periods) - split_idx,
        )

        inner_train, val_data = train_test_split(train_data, split_period)

        ests: list[Any] = []
        for tmpl in self.base_templates:
            est_cls = cast("type[Any]", tmpl.get_model(None))
            ests.append(est_cls())
        preds_inner = [e.train(inner_train) for e in ests]

        df_val = val_data.to_pandas()
        y_val = df_val[self.target_col].to_numpy()
        key_cols = ["location", "time_period"]

        meta_list: list[np.ndarray] | None = None
        meta_mat: np.ndarray | None = None
        if self.method == "probabilistic":
            meta_list = [
                _SampleExtractor.reshape_samples(
                    p.predict(inner_train, val_data),
                    df_val,
                    self.n_samples,
                    rng=rng,  # keep a shared RNG for reproducibility
                )
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
        s = float(np.sum(coef))
        if s <= 0:
            logger.warning("Non-positive meta-model weights; falling back to uniform weights")
            self.weights = np.full(len(coef), 100.0 / len(coef))
        else:
            self.weights = coef / s * 100.0

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
            rng=rng,  # forward the shared RNG
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
