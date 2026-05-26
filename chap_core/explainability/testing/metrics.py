"""Faithfulness metric (eLoss) for LIME explanations.

Implements the ``delta eLoss`` faithfulness metric introduced by Nguyen, Le
Nguyen and Ifrim and adapted for chap-core in Leander Skoglund's MSc thesis:

- Sort the explanation's features by absolute coefficient.
- For each k in deciles of ``num_features``, perturb the top-k most important
  features (using the same sampler the explanation was generated with) and
  measure ``|y_perturbed - y_orig|``. Build the same curve for the bottom-k
  least important features.
- Trapezoidal AUC of each curve, then ``delta = AUC(top-k) - AUC(bottom-k)``.

A higher ``delta`` means perturbing the features the explanation flagged as
important changes the prediction more than perturbing the features it
flagged as unimportant, i.e. the explanation is more faithful to the
black-box model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from chap_core.explainability.perturb import SampleModel
    from chap_core.explainability.segment import Indices
    from chap_core.models.external_model import ExternalModel
    from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


def eLoss(
    *,
    model: ExternalModel,
    original_vector: dict,
    feature_map: list[tuple[str, str, int | None]],
    sorted_explanation: list[tuple[str, float]],
    sampler: SampleModel,
    hist_df: pd.DataFrame,
    fut_df: pd.DataFrame,
    feature_names: list[str],
    features_hist: list[str],
    features_fut: list[str],
    horizon: int,
    location: str,
    hist_type: type | None,
    fut_type: type | None,
    feat_indices: dict[str, Indices],
    y_orig: float,
    full_dataset: DataSet | None,
    full_future_weather: DataSet | None,
    n_buckets: int = 10,
) -> tuple[float, float, float]:
    """Compute ``(delta_eloss, auc_top, auc_bottom)`` for a LIME explanation.

    Args:
        model: The black-box model the explanation was generated for.
        original_vector: Original feature vector that produced ``y_orig``.
        feature_map: Output of ``build_feature_map`` — list of
            ``(name, parent_key, lag)`` triples in mask order.
        sorted_explanation: ``(feature_name, coefficient)`` pairs sorted by
            absolute coefficient descending (i.e. the surrogate's importance
            ranking).
        sampler: The same sampler used to generate the explanation; used here
            to replace turned-off features.
        hist_df / fut_df: Historical and future feature dataframes for the
            location being explained.
        feature_names / features_hist / features_fut: Name lists used by
            ``perturb_vectors`` / ``produce_lime_dataset``.
        horizon / location / hist_type / fut_type / feat_indices: Forwarded
            unchanged to the perturbation + prediction machinery.
        y_orig: Baseline model prediction for the unperturbed input.
        full_dataset / full_future_weather: Optional surrounding context the
            model may need when ``model.predict`` is one-hot-location-sensitive
            (matching ``produce_lime_dataset``'s fallback path).
        n_buckets: How many k-values to sample across ``[1, num_features]``.
            Defaults to 10 (deciles), matching the thesis.

    Returns:
        Tuple ``(delta_eloss, auc_top, auc_bottom)``. ``delta_eloss`` is the
        primary metric; positive values indicate a faithful explanation.
    """
    # Lazy import to avoid a circular dependency: lime.py imports this module
    # inside its `if return_metrics:` block at call time.
    from chap_core.explainability.lime import perturb_vectors, produce_lime_dataset

    num_features = len(feature_names)
    if num_features == 0:
        return 0.0, 0.0, 0.0

    feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    importance_order = [name for name, _ in sorted_explanation if name in feature_to_idx]
    if not importance_order:
        return 0.0, 0.0, 0.0

    # Pick k at the requested buckets across [1, num_features], deduped.
    k_values: list[int] = sorted({max(1, round(num_features * i / n_buckets)) for i in range(1, n_buckets + 1)})

    def _build_masks(off_groups: list[list[str]]) -> list[np.ndarray]:
        masks: list[np.ndarray] = []
        for off in off_groups:
            mask = np.ones(num_features)
            for fname in off:
                mask[feature_to_idx[fname]] = 0
            masks.append(mask)
        return masks

    top_masks = _build_masks([importance_order[:k] for k in k_values])
    bottom_masks = _build_masks([importance_order[-k:] for k in k_values])

    def _deviation_curve(masks: list[np.ndarray]) -> np.ndarray:
        pb, pb_mask = perturb_vectors(hist_df, original_vector, feat_indices, sampler, feature_map, masks)
        _, y, _, _ = produce_lime_dataset(
            model,
            hist_df,
            fut_df,
            pb,
            pb_mask,
            feature_names,
            features_hist,
            features_fut,
            horizon,
            location,
            feat_indices,
            hist_type,
            fut_type,
            full_dataset=full_dataset,
            full_future_weather=full_future_weather,
        )
        return np.asarray(np.abs(y - y_orig))

    top_dev = _deviation_curve(top_masks)
    bottom_dev = _deviation_curve(bottom_masks)

    k_array = np.asarray(k_values, dtype=float)
    auc_top = float(np.trapezoid(top_dev, k_array))
    auc_bottom = float(np.trapezoid(bottom_dev, k_array))
    delta = auc_top - auc_bottom
    return delta, auc_top, auc_bottom
