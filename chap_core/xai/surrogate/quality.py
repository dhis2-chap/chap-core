from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import r2_score

from .registry import get_display_name


@dataclass
class SurrogateQuality:
    r_squared: float | None = None
    mae: float = 0.0
    mape: float | None = None
    n_samples: int = 0
    n_unique_rows: int = 0
    constant_features: list[str] = field(default_factory=list)
    imputation_rates: dict[str, float] = field(default_factory=dict)
    removed_features: list[str] = field(default_factory=list)
    selected_model_type: str | None = None
    selected_model_display_name: str | None = None
    n_groups: int | None = None
    fidelity_tier: str = "good"
    residual_mean: float | None = None
    residual_std: float | None = None
    target_transformed: bool = False
    target_transform_method: str | None = None
    permutation_removed_features: list[str] = field(default_factory=list)
    r_squared_train: float | None = None
    generalization_gap: float | None = None

    def to_dict(self) -> dict:
        def _r(v: float | None, ndigits: int) -> float | None:
            return round(v, ndigits) if v is not None else None

        return {
            "r_squared": _r(self.r_squared, 6),
            "mae": round(self.mae, 4),
            "mape": _r(self.mape, 4),
            "n_samples": self.n_samples,
            "n_unique_rows": self.n_unique_rows,
            "constant_features": self.constant_features,
            "imputation_rates": {k: round(v, 4) for k, v in self.imputation_rates.items()},
            "removed_features": self.removed_features,
            "selected_model_type": self.selected_model_type,
            "selected_model_display_name": self.selected_model_display_name,
            "n_groups": self.n_groups,
            "fidelity_tier": self.fidelity_tier,
            "residual_mean": _r(self.residual_mean, 6),
            "residual_std": _r(self.residual_std, 6),
            "target_transformed": self.target_transformed,
            "target_transform_method": self.target_transform_method,
            "permutation_removed_features": self.permutation_removed_features,
            "r_squared_train": _r(self.r_squared_train, 6),
            "generalization_gap": _r(self.generalization_gap, 6),
        }


def _compute_fidelity_tier(r2: float | None) -> str:
    if r2 is None or r2 < 0.5:
        return "poor"
    if r2 < 0.8:
        return "moderate"
    return "good"


def compute_surrogate_quality(
    X_filtered: np.ndarray,
    y: np.ndarray,
    model: Any,
    model_type: str,
    groups: np.ndarray | None,
    loo_preds: np.ndarray,
    r2_cv: float | None,
    imputation_rates: dict[str, float],
    constant_features: list[str],
    removed_features: list[str],
    perm_removed_features: list[str],
    target_transform_method: str | None,
    n_samples: int,
) -> SurrogateQuality:
    """Assemble a SurrogateQuality from a fitted model and LOO evaluation results.

    *n_samples* is the row count of the original (pre-filter) feature matrix and is
    stored as-is in SurrogateQuality for reporting purposes.
    """
    train_preds = model.predict(X_filtered)
    ref_preds = loo_preds if r2_cv is not None else train_preds
    errors = np.abs(y - ref_preds)
    mae = float(np.mean(errors))
    nonzero = np.abs(y) > 1e-8
    mape: float | None = float(np.mean(errors[nonzero] / np.abs(y[nonzero]))) if nonzero.any() else None

    residual_mean: float | None = None
    residual_std: float | None = None
    if r2_cv is not None:
        residuals = y - loo_preds
        residual_mean = float(np.mean(residuals))
        residual_std = float(np.std(residuals))

    n_groups: int | None = len(np.unique(groups)) if groups is not None else None
    unique_rows = len(np.unique(X_filtered, axis=0))

    ss_tot_train = float(np.sum((y - np.mean(y)) ** 2))
    r2_train: float | None = float(r2_score(y, train_preds)) if ss_tot_train > 0 else None

    generalization_gap: float | None = r2_train - r2_cv if (r2_train is not None and r2_cv is not None) else None

    return SurrogateQuality(
        r_squared=r2_cv,
        mae=mae,
        mape=mape,
        n_samples=n_samples,
        n_unique_rows=unique_rows,
        constant_features=constant_features,
        imputation_rates=imputation_rates,
        removed_features=removed_features,
        selected_model_type=model_type,
        selected_model_display_name=get_display_name(model_type),
        n_groups=n_groups,
        fidelity_tier=_compute_fidelity_tier(r2_cv),
        residual_mean=residual_mean,
        residual_std=residual_std,
        target_transformed=target_transform_method is not None,
        target_transform_method=target_transform_method,
        permutation_removed_features=perm_removed_features,
        r_squared_train=r2_train,
        generalization_gap=generalization_gap,
    )
