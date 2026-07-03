# Modeling internals: in-tree vs external ML

A recurring question is whether `chap_core` contains real ML code, or whether it
is "just data preparation and orchestration" for models that live elsewhere. The
honest answer is nuanced, and this page draws the boundary explicitly.

## TL;DR

`chap_core` is primarily an **orchestration + evaluation framework**. The actual
forecasting models — the ones that learn disease dynamics from data — live in
**external model repositories** and run through a `TrainPredictRunner` or a chapkit
HTTP service (see [Architecture diagrams](architecture_diagrams.md)).

The ML that *does* live inside `chap_core` is **classical/statistical only**. There
is no deep learning in the package. Confirmed absent from `chap_core`: `torch`,
`tensorflow`, `jax`, `statsmodels`, `xgboost`, `lightgbm`, `pymc`, `prophet`,
`numpyro`, `pyro`, `gpflow`.

The direct ML-relevant dependencies `chap_core` actually uses:

| Library | Role in chap_core |
|---|---|
| scikit-learn | climate predictor, Poisson baselines, KMeans clustering, LIME surrogate |
| optuna | hyperparameter search (HPO) |
| gluonts | evaluation (`Evaluator`) and data adaptors |
| scipy | explainability (signal / distance helpers) |
| numpy / pandas | all hand-written scoring metrics |

## Three categories

It helps to split the in-tree code into three buckets, because much of what looks
like "ML in chap_core" is actually tooling around models, or evaluation statistics.

| Category | What it is | Examples |
|---|---|---|
| A | Genuinely ML — learns from data to predict | climate predictor, Poisson/naive baselines, KMeans |
| B | ML *tooling* — uses ML, but is not itself a forecaster | HPO (optuna), LIME explainability |
| C | Not ML — evaluation statistics or heuristics | CRPS/Winkler/MAE metrics, preference learning |

### A — Genuinely ML (learns to predict)

- **`chap_core/climate_predictor.py`** — fits one scikit-learn `LinearRegression`
  per `(location, climate field)` on a one-hot month/week feature matrix. This is
  the one in-tree model that is truly load-bearing: it **fabricates the future
  climate covariates** that a disease model needs before it can predict (the disease
  model never receives ground-truth future weather). It runs in both the
  real-prediction path (via `forecast_with_predicted_weather`) and the backtesting
  path (via `QuickForecastFetcher`).
- **`chap_core/predictor/poisson.py`, `chap_core/predictor/naive_predictor.py`** —
  scikit-learn `PoissonRegressor` baselines (including a per-location, lagged-cases +
  one-hot-season variant). Mostly used in tests and example models.
- **`chap_core/predictor/naive_estimator.py`** — `NaiveEstimator`, the only built-in
  model wired into the live system: the database special-cases the configured model
  named `naive_model` and returns it as a fast, R-free stand-in. It predicts the
  per-location mean and draws Poisson samples — barely ML (essentially a statistic).
- **`chap_core/feature_generators/seasonality_cluster.py`** and
  **`chap_core/plotting/season_plot.py`** — scikit-learn `KMeans` seasonality
  clustering (unsupervised).

### B — ML tooling (uses ML, but is not a forecaster)

- **`chap_core/hpo/`** — hyperparameter optimization built on optuna's TPE sampler
  (plus hand-rolled grid and random searchers). It tunes an *external* model
  template's hyperparameters against a backtest metric (default RMSE). It is
  **CLI-only** and not exposed in the REST API:

  ```console
  chap eval --estimator-options.mode=hpo ...
  ```

- **`chap_core/explainability/`** (`lime.py`, `surrogate.py`, `distance.py`) —
  post-hoc LIME explanations. It fits a scikit-learn `Ridge` local surrogate around
  an already-trained model to report feature importance. Note the nuance: LIME uses
  ML mechanics (it fits a regression) but is **not a predictive model** — it is
  interpretability tooling *about* a model, so under a strict "learns-to-predict"
  definition it does not count as an ML model.

### C — Not ML (evaluation statistics / heuristics)

- **`chap_core/assessment/metrics/`** — the scoring metrics, all hand-implemented on
  numpy: a from-scratch order-statistic **CRPS** estimator, Winkler interval scores,
  MAE / RMSE / MAPE, percentile-coverage calibration, outbreak-detection
  sensitivity/specificity, and peak-timing diff. This is substantive statistical
  code and the busiest ML-adjacent surface (the REST worker, the REST API router, the
  CLI, and HPO all route through it) — but it is *scoring*, not learning. See
  [Evaluation pipeline](evaluation_pipeline.md) and
  [Creating custom metrics](creating_custom_metrics.md).
- **`chap_core/preference_learning/`** — despite a docstring mentioning Bayesian /
  bandit strategies, the only implementation is a plain tournament bracket that
  A/B-compares model configurations. No ML library; one CLI command. See
  [Preference learning](preference_learning.md).
- gluonts `Evaluator` usage in **`chap_core/assessment/prediction_evaluator.py`** —
  quantile-based evaluation plumbing, not a model.

## The external boundary

Everything with real modeling weight lives outside `chap_core`. The production
forecasting models (for example the DeepAR-based autoregressive models, and R/INLA
models) are maintained in their own repositories and executed through a
`TrainPredictRunner` implementation (Docker, UV, Conda, Renv, command-line, MLflow)
or against a remote chapkit HTTP service. The plumbing that loads and runs them is
documented in [Code overview](code_overview.md), and the reference/example models
are catalogued in
[Running models in Chap](../external_models/running_models_in_chap.md).

The two entry points into this shared core — the `chap` CLI and the REST API — are
compared in [CLI vs REST API](cli_vs_rest_api.md).

## One-line answer

The only strictly-ML (learns-to-predict) code in `chap_core` is the scikit-learn
climate-covariate predictor and the Poisson/naive baselines, plus KMeans clustering.
HPO and LIME are ML *tooling*; the metrics and preference learning are statistics and
heuristics. All heavy and deep-learning modeling is external.
