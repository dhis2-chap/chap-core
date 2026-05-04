"""Optuna-based hyperparameter tuning for surrogate models."""

import logging
from typing import Any

import numpy as np

from .model import build_surrogate_model, resolve_model_params
from .registry import DEFAULT_MODEL_TYPE, get_model_info

logger = logging.getLogger(__name__)

__all__ = [
    "select_and_tune_best_model_type",
    "tune_surrogate_hyperparameters",
]


class _PatienceCallback:
    """Stop an Optuna study after *patience* consecutive completed trials with no improvement.

    Create a new instance per study; do not reuse across multiple calls to ``study.optimize``.
    """

    def __init__(self, patience: int = 20) -> None:
        self._patience = patience
        self._best: float = float("-inf")
        self._no_improve: int = 0

    def __call__(self, study: Any, trial: Any) -> None:
        import optuna

        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        if study.best_value > self._best:
            self._best = study.best_value
            self._no_improve = 0
        else:
            self._no_improve += 1
        if self._no_improve >= self._patience:
            study.stop()


def _suggest_param(trial, name: str, spec: dict[str, Any], n_samples: int) -> Any:
    """Map a tunable_param spec entry to an Optuna trial suggestion."""
    kind = spec["type"]
    low = spec["low"]
    high = spec.get("high")
    if high is None:
        fraction = spec.get("high_n_fraction", 5)
        high = max(low, n_samples // fraction)
    if n_samples <= 80:
        if name == "min_samples_leaf":
            low = max(int(low), 2)
        elif name == "min_samples_split":
            low = max(int(low), 4)
    if kind == "int":
        return trial.suggest_int(name, low, high)
    if kind == "float":
        return trial.suggest_float(name, low, high, log=spec.get("log", False))
    raise ValueError(f"Unknown tunable_param type '{kind}' for param '{name}'")


def _run_tuning_study(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    n_trials: int,
    groups: np.ndarray | None,
    random_state: int,
) -> tuple[dict[str, Any], float]:
    """Run an Optuna study for *model_type* and return ``(best_params, best_cv_score)``.

    The CV score is ``-MSE / Var(y)`` (scale-invariant, ≈ ``-(1 - R²)``), averaged
    across folds. The optimisation objective also penalizes fold-score variance and
    train-vs-validation overfitting gap to improve generalization stability on
    small datasets. ``GroupKFold`` is used when *groups* has ≥ 2 unique values.
    """
    import optuna
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import GroupKFold, KFold

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X, y = np.asarray(X), np.asarray(y)
    info = get_model_info(model_type)
    tunable = info["tunable_params"]
    n_samples = len(X)

    y_var = float(np.var(y)) + 1e-8

    effective_groups: np.ndarray | None = None
    if groups is not None and len(np.unique(groups)) >= 2:
        effective_groups = groups
        n_unique = len(np.unique(groups))
        cv_splitter = GroupKFold(n_splits=min(n_unique, 5))
    else:
        n_splits = min(5, max(2, n_samples // 3)) if n_samples >= 6 else max(2, min(n_samples, 5))
        n_splits = min(n_splits, n_samples)
        if n_splits < 2:
            n_splits = 2
        cv_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    n_features = int(X.shape[1]) if X.ndim == 2 else 1
    low_sample_regime = n_samples <= 80
    stability_weight = 0.18 if low_sample_regime else 0.08
    overfit_weight = 0.25 if low_sample_regime else 0.12
    if model_type == "gradient_boosting" and low_sample_regime:
        stability_weight = 0.22
        overfit_weight = 0.35
    if n_features <= 6:
        overfit_weight += 0.05

    def objective(trial: optuna.Trial) -> float:
        trial_params = {name: _suggest_param(trial, name, spec, n_samples) for name, spec in tunable.items()}
        if model_type == "xgboost":
            trial_params.pop("early_stopping_rounds", None)
        params = resolve_model_params(model_type, {}, trial_params)

        def model_template():
            return build_surrogate_model(model_type, params, random_state=random_state, n_samples=n_samples)

        split_kwargs = {"groups": effective_groups} if effective_groups is not None else {}
        splits = list(cv_splitter.split(X, y, **split_kwargs))
        val_scores: list[float] = []
        train_scores: list[float] = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            m = model_template()
            m.fit(X[train_idx], y[train_idx])
            val_score = -mean_squared_error(y[test_idx], m.predict(X[test_idx])) / y_var
            train_score = -mean_squared_error(y[train_idx], m.predict(X[train_idx])) / y_var
            val_scores.append(val_score)
            train_scores.append(train_score)
            current_mean = float(np.mean(val_scores))
            current_var_penalty = stability_weight * float(np.std(val_scores)) if len(val_scores) > 1 else 0.0
            current_overfit_penalty = overfit_weight * float(
                np.mean(np.maximum(0.0, np.asarray(train_scores) - np.asarray(val_scores)))
            )
            trial.report(current_mean - current_var_penalty - current_overfit_penalty, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned

        mean_val = float(np.mean(val_scores))
        var_penalty = stability_weight * float(np.std(val_scores)) if len(val_scores) > 1 else 0.0
        mean_gap = float(np.mean(np.maximum(0.0, np.asarray(train_scores) - np.asarray(val_scores))))
        if mean_gap > 0.5:
            return float("-inf")
        overfit_penalty = overfit_weight * mean_gap
        return mean_val - var_penalty - overfit_penalty

    n_startup = max(10, min(30, n_trials // 4))
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state, n_startup_trials=n_startup),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=n_startup, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[_PatienceCallback(patience=20)])
    return study.best_params, study.best_value


def _score_fixed_params(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    params: dict[str, Any],
    groups: np.ndarray | None,
    random_state: int,
) -> float:
    """Score a fixed param set via CV with a given random_state. Returns mean -MSE/Var(y).

    Uses plain (unpenalized) score intentionally so the 3x3 cross-evaluation in
    ``_run_multi_seed_tuning`` ranks candidates on neutral predictive performance,
    not on a seed-dependent variance penalty.
    """
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import GroupKFold, KFold

    X, y = np.asarray(X), np.asarray(y)
    n_samples = len(X)
    y_var = float(np.var(y)) + 1e-8

    effective_groups: np.ndarray | None = None
    if groups is not None and len(np.unique(groups)) >= 2:
        effective_groups = groups
        n_unique = len(np.unique(groups))
        cv_splitter = GroupKFold(n_splits=min(n_unique, 5))
    else:
        n_splits = min(5, max(2, n_samples // 3)) if n_samples >= 6 else max(2, min(n_samples, 5))
        n_splits = min(n_splits, n_samples)
        if n_splits < 2:
            n_splits = 2
        cv_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    split_kwargs = {"groups": effective_groups} if effective_groups is not None else {}
    splits = list(cv_splitter.split(X, y, **split_kwargs))

    resolved = resolve_model_params(model_type, {}, params)
    val_scores: list[float] = []
    for train_idx, test_idx in splits:
        m = build_surrogate_model(model_type, resolved, random_state=random_state, n_samples=n_samples)
        m.fit(X[train_idx], y[train_idx])
        val_scores.append(-mean_squared_error(y[test_idx], m.predict(X[test_idx])) / y_var)

    return float(np.mean(val_scores))


_MULTI_SEED_SEEDS = (42, 123)


def _run_multi_seed_tuning(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    n_trials: int,
    groups: np.ndarray | None,
    seeds: tuple[int, ...] = _MULTI_SEED_SEEDS,
) -> tuple[dict[str, Any], float]:
    """Run N seeded Optuna studies, cross-evaluate all best-param candidates, return the one
    with the best median score across seeds.

    For small datasets (n <= 80) where a single random seed may yield unstable results,
    this selects params that generalise consistently rather than just performing well on
    one lucky CV split.
    """
    candidates: list[dict[str, Any]] = []
    for seed in seeds:
        params, _ = _run_tuning_study(X, y, model_type, n_trials, groups, seed)
        candidates.append(params)

    scores_matrix = [[_score_fixed_params(X, y, model_type, p, groups, seed) for seed in seeds] for p in candidates]
    medians = [float(np.median(row)) for row in scores_matrix]
    best_idx = int(np.argmax(medians))
    logger.info(
        "Multi-seed tuning (%s): candidate medians=%s, selected candidate %d",
        model_type,
        [round(m, 4) for m in medians],
        best_idx,
    )
    return candidates[best_idx], medians[best_idx]


def tune_surrogate_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = DEFAULT_MODEL_TYPE,
    n_trials: int = 30,
    groups: np.ndarray | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Use Optuna to find optimal hyperparameters for the given surrogate model type.

    Uses ``neg_mean_squared_error`` as the Optuna objective (more stable than R²
    for optimisation on small datasets).  When ``groups`` is provided and has at
    least 2 unique values, uses ``GroupKFold`` so that all rows from the same
    org_unit stay in the same fold.

    For ``n_samples <= 80``, runs multi-seed tuning and returns the most stable params.

    Returns the best params dict found.
    """
    if len(X) <= 80:
        params, _ = _run_multi_seed_tuning(X, y, model_type, n_trials, groups)
    else:
        params, _ = _run_tuning_study(X, y, model_type, n_trials, groups, random_state)
    return params


def select_and_tune_best_model_type(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    n_trials: int = 80,
    random_state: int = 42,
) -> tuple[str, dict[str, Any]]:
    """Select the best surrogate model type and its tuned hyperparameters.

    Combines LOO-based candidate ranking with Optuna tuning so that the final
    winner is chosen on the same CV metric used for fitting, not on cheap LOO
    proxies alone.

    Steps:
      1. Run ``auto_select_best_model_type`` to get up to 3 LOO-ranked candidates.
      2. Tune each candidate with ``n_trials // n_candidates`` trials (floor: 20).
      3. Return ``(model_type, best_params)`` for the candidate with the highest
         tuned CV score.

    For ``n_samples <= 80``, each candidate is tuned with multi-seed tuning for
    more stable hyperparameter selection.

    Falls back to the LOO winner with empty params when tuning fails for all
    candidates.
    """
    from .model import auto_select_best_model_type

    candidates = auto_select_best_model_type(X, y, groups=groups, random_state=random_state)
    n_candidates = min(3, len(candidates))
    candidates = candidates[:n_candidates]
    trials_per = max(20, n_trials // n_candidates)
    use_multi_seed = len(X) <= 80

    best_model_type = candidates[0]
    best_params: dict[str, Any] = {}
    best_score = float("-inf")

    for model_type in candidates:
        try:
            if use_multi_seed:
                params, score = _run_multi_seed_tuning(X, y, model_type, trials_per, groups)
            else:
                params, score = _run_tuning_study(X, y, model_type, trials_per, groups, random_state)
            logger.info("Candidate %s tuned CV score=%.4f", model_type, score)
            if score > best_score:
                best_score = score
                best_model_type = model_type
                best_params = params
        except Exception as exc:
            logger.warning("Candidate %s tuning failed: %s", model_type, exc)

    logger.info("Selected surrogate: %s (tuned CV score=%.4f)", best_model_type, best_score)
    return best_model_type, best_params
