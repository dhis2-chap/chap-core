# XAI Flow in CHAP

This page explains how CHAP generates explainability outputs for a prediction, in a simple end-to-end flow.

## Big picture

For each prediction, CHAP can explain results in two ways:

- **Native SHAP**: use SHAP values produced directly by the model (see [Native SHAP](native_shap.md)).
- **Surrogate-based XAI**: train a separate tabular model that mimics the prediction outputs, then run SHAP or LIME on that surrogate.

Available methods:

| Method | Description |
|--------|-------------|
| `shap_auto` | Auto-select the best surrogate family, then compute SHAP |
| `shap_xgboost` | XGBoost surrogate + SHAP |
| `shap_lightgbm` | LightGBM surrogate + SHAP |
| `shap_hist_gradient_boosting` | HistGradientBoosting surrogate + SHAP |
| `shap_random_forest` | RandomForest surrogate + SHAP |
| `shap_gradient_boosting` | GradientBoosting surrogate + SHAP |
| `shap_extra_trees` | ExtraTrees surrogate + SHAP |
| `lime_auto` | Auto-select the best surrogate family, then compute LIME |
| `native_shap` | Model-native SHAP values (no surrogate fitted) |

`shap_auto` and `lime_auto` benchmark the available surrogate families and pick the one with the highest LOO-R² before computing attributions.

## End-to-end flow

1. A prediction is stored with forecast rows (`org_unit`, `period`, forecast samples).
2. The XAI endpoint is called for global, local, beeswarm, or horizon explanations.
3. CHAP loads the source dataset and builds a surrogate training matrix:
   - `X`: covariate values matched to each forecast row.
   - `y`: one scalar target per forecast row (median by default, or mean/quantile).
   - `groups`: org-unit ids for grouped cross-validation.
4. If the method is `native_shap` and native SHAP data exists for this prediction, CHAP reads stored values directly (no surrogate fitting).
5. Otherwise CHAP trains a surrogate model (or reuses a cached one for the same prediction/method/output statistic).
6. CHAP computes explanations:
   - **Global**: top features across rows.
   - **Local**: feature attribution for one row (`org_unit`, `period`).
   - **Beeswarm**: many local attributions packed into one response.
   - **Horizon summary**: local attributions grouped over forecast steps.
7. CHAP stores explanation outputs in `PredictionExplanation` rows (local) and prediction metadata (global), then serves them through API responses.

## How surrogate training data is built

CHAP creates one training row per forecast:

- Finds matching covariate row by `org_unit` and `period`.
- Supports period fallback logic for horizon-style periods like `YYYYMM_k`:
  1. Exact period match.
  2. Historical same-month (or same-week) mean over prior years.
  3. Last available row for that org unit.
- Builds missing-value statistics per feature.
- Imputes missing covariates with feature medians.
- Computes `y` from forecast samples:
  - `median` (default),
  - `mean`, or
  - `qXX` quantile.

This gives the supervised dataset used for surrogate fitting.

## Supported surrogate model families

CHAP ships with these surrogate families:

| Family | Type | Notes |
|--------|------|-------|
| `hist_gradient_boosting` | tree | Default when auto-select is used |
| `xgboost` | tree | Optional dependency |
| `lightgbm` | tree | Optional dependency |
| `gradient_boosting` | tree | |
| `random_forest` | tree | |
| `extra_trees` | tree | |
| `decision_tree` | tree | |
| `catboost` | tree | Optional dependency |
| `ridge` | linear | |
| `lasso` | linear | |

Tree surrogates use `shap.TreeExplainer`; linear surrogates use `shap.LinearExplainer` with the training data as background.

## How SHAP surrogate models are trained

For `shap_*` methods:

1. **Feature filtering**
   - Remove constant features.
   - Remove features with very high imputation rate (>= 90%).
   - On larger datasets (≥100 samples, >6 features), optionally remove weak/noisy features with permutation importance, retaining at least 3 features.
2. **Model selection**
   - If method is fixed (for example `shap_xgboost`), use that model type.
   - If method is `shap_auto`, benchmark all available surrogate families using LOO-R² and keep the top candidates.
3. **Hyperparameter tuning**
   - For enough samples, run Optuna tuning with cross-validation.
   - For `shap_auto`, tune candidate models and keep the best tuned model.
4. **Target transformation**
   - CHAP evaluates whether applying `log1p` or Yeo-Johnson to the target (`y`) improves CV fit.
   - If a transform is selected, the surrogate is wrapped with `TransformedTargetRegressor`; SHAP values are inverse-scaled back to the original prediction space before being returned.
5. **Final fit**
   - Fit the selected surrogate on filtered features.
6. **Quality scoring**
   - Compute fidelity metrics: LOO-R², train R², MAE, residual statistics.
   - Assign a fidelity tier:
     - **good**: R² ≥ 0.8 — surrogate closely mimics the model.
     - **moderate**: 0.5 ≤ R² < 0.8 — notable unexplained variance.
     - **poor**: R² < 0.5 — surrogate does not mimic the model well; treat attributions with caution.

After training, CHAP computes SHAP values from the surrogate and maps them back to the full feature list (removed features get zero attribution).

About SHAP computation:

- Tree-based surrogates use `shap.TreeExplainer`, which computes exact Shapley values from split structure without permuting features at runtime.
- Linear surrogates use `shap.LinearExplainer` with the training set as background.
- If a SHAP explainer cannot be constructed (missing optional dependency, unusual model), CHAP falls back to occlusion-based local attribution.

## How LIME surrogate models are trained

`lime_auto` uses full auto-selection — it benchmarks all available surrogate families by LOO-R² and picks the best, identical to `shap_auto`. The training path is:

1. Build the same `(X, y, groups)` data.
2. Run the same feature filtering.
3. Benchmark all available surrogate families by LOO-R² and select the best.
4. Run the same Optuna tuning when data size allows.
5. Apply the same target transformation evaluation.
6. Fit the final surrogate model.

The difference from SHAP methods is only in attribution:

- **SHAP methods** use `shap.TreeExplainer` or `shap.LinearExplainer` on the fitted surrogate to compute exact or approximate Shapley values.
- **LIME methods** wrap the surrogate's `predict` function in `LimeTabularExplainer`, generate a perturbed neighborhood around the instance, fit a local linear model to those perturbations, and read off local feature contributions.

For global LIME explanations, CHAP aggregates local LIME importances across up to 50 instances and ranks features by average absolute attribution. For global SHAP explanations, CHAP aggregates exact SHAP values across all rows.

So SHAP and LIME share the same surrogate training pipeline; they differ only in how they extract attributions from the trained surrogate.

Why LIME is not run directly on the original model:

- CHAP's prediction interfaces are heterogeneous (Python, R, Docker, remote services), and many do not expose a stable, low-latency tabular `predict` callable needed for LIME neighborhood sampling.
- LIME requires many repeated local evaluations; calling the original model repeatedly can be expensive, non-deterministic, or operationally fragile.
- The fitted surrogate provides a consistent, fast prediction surface aligned with the stored forecast target (`median`, `mean`, or `qXX`), so both SHAP and LIME operate on the same explanation target and are easier to compare.

Important limitation of surrogate-based explanations:

- Surrogate attributions explain how the surrogate maps the tabular covariates to the prediction target, not how the original model internally represented or transformed those inputs.
- If the original model applies hidden feature extraction, embeddings, nonlinear transformations, or other internal preprocessing, that internal feature logic is not directly visible in surrogate explanations. The surrogate is trained only on the tabular covariates CHAP can see, so it may fit the forecast target more poorly and represent the original model less faithfully than when the original mapping is closer to those inputs.
- This gap is a key reason CHAP also supports `native_shap`: when model-native SHAP values are available, they provide attributions from the original model itself without relying on surrogate approximation.

## Caching and persistence

- Trained surrogates are cached in memory per key `(prediction_id, xai_method_name, output_statistic)` (FIFO eviction, up to 20 entries) to avoid repeated fits in the same process.
- Local explanations are persisted in the `PredictionExplanation` table.
- Global summaries are persisted in prediction metadata under method-specific keys.

This is why repeated calls for the same inputs are usually much faster.
