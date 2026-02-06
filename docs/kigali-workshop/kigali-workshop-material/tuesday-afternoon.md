# Tuesday Afternoon - 24 Feb

## Design Document: CLIM-442 - Prepare Tuesday Afternoon Session

### Context

This session follows Tuesday morning (which covers evaluation) and Monday afternoon
(evaluation walkthrough). By this point, participants have built a minimalist model,
gotten it running in CHAP, and evaluated it. The afternoon session introduces
progressively more complex modeling effects, culminating in multi-region strategies.

### Session Goals

- Introduce modeling effects in a progressive order, building complexity step by step
- Introduce how to handle multiple regions and the rationale for borrowing information across regions
- Have participants extend their own model with these effects

### Order of Effects Introduction

Effects are introduced in this order:

1. **Location-specific offset** -- the simplest way to make a model region-aware
2. **Seasonal effect** -- periodic patterns (month-of-year, Fourier terms)
3. **Lagged covariates** -- past climate variables as predictors
4. **Lagged target** -- past disease cases as predictor (introduced late because of
   technical difficulty in evaluation setup, even though it is typically the most
   important predictor)
5. **Interactions between location and effects** -- location-specific slopes,
   borrowing strength across regions

### BasicEstimator

To support the walkthroughs, create a `BasicEstimator` class similar to
`NaiveEstimator`. The key difference: `BasicEstimator` takes a feature extraction
function as a constructor argument. This function receives a DataFrame and returns
the feature matrix used for fitting/prediction. This lets the walkthroughs
progressively build complexity by swapping in different feature extraction functions
while reusing the same estimator infrastructure.

```
estimator = BasicEstimator(extract_features=my_feature_fn)
predictor = estimator.train(dataset)
predictions = predictor.predict(historic_data, future_data)
```

The feature extraction function signature:

```
def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract feature columns from a CHAP DataFrame."""
    ...
```

This keeps each walkthrough step focused on the modeling idea (what features to
extract) rather than boilerplate.

`BasicEstimator` should follow the same two-phase pattern as `NaiveEstimator`:
- `train(data: DataSet) -> BasicPredictor` -- fits a model using extracted features
- `BasicPredictor.predict(historic_data, future_data) -> DataSet[Samples]`
- Class attributes `model_template_db` and `configured_model_db` for evaluation
  integration
- `save()`/`load()` for serialization

### Existing Material

The `learn_modelling.md` page already contains substantial written material on
multiple regions, lags, borrowing strength, and bias-variance tradeoff, with links
to external tutorial repos (`minimalist_multiregion`, `minimalist_example_lag`).

What is missing: doctested Python walkthroughs that participants can follow
interactively within the docs, similar to the evaluation walkthrough.

### Plan

#### 1. Implement BasicEstimator (prerequisite)

Create `chap_core/predictor/basic_estimator.py` with:

- `BasicEstimator(extract_features: Callable)` -- constructor takes feature fn
- Fits a linear regression (or similar simple model) on extracted features
- Returns `BasicPredictor` that generates `Samples` via prediction + noise model
- Includes `model_template_db` and `configured_model_db` class attributes
- Add tests following existing `test_naive_estimator.py` pattern

#### 2. Doctested walkthrough: regression with progressive effects (CLIM-477)

Create a documentation page with executable code blocks showing:

1. Load data, introduce `BasicEstimator`
2. Location-specific offset as feature (intercept per region)
3. Add seasonal effect (month-of-year or Fourier terms)
4. Add lagged climate covariates (rainfall, temperature at various lags)
5. Add lagged target (past disease cases) -- note the technical difficulty:
   at prediction time, future disease cases are unknown, so the model must
   use its own predictions recursively or only use lags beyond the forecast horizon
6. Evaluate each variant to show incremental improvement

#### 3. Doctested walkthrough: multi-region modeling strategies (CLIM-478)

Create a documentation page with executable code blocks covering:

- Separate model per region (independent fits)
- Single global model (all regions pooled, ignoring region identity)
- Model with shared fixed effect but separate seasonal effect per region
- Partial pooling / hierarchical model

Each approach evaluated side-by-side so participants can compare prediction skill.

#### 4. Overview of location specificity in common model classes (CLIM-479)

Create a reference table based on common models from the literature, covering
how each handles location-specific vs shared effects:

| Model class | Separate per region | Global (shared) | Semi-global / partial pooling |
|---|---|---|---|
| Linear regression | Yes (fit per region) | Yes (pooled) | Via mixed effects |
| ARIMA | Yes (standard use) | Not typical | Not typical |
| ETS | Yes (standard use) | Not typical | Not typical |
| Hierarchical Bayesian | Yes | Yes | Yes (primary use case) |
| Random Forest / XGBoost | Yes | Yes (with region feature) | Via region feature |
| Deep learning (LSTM, etc.) | Yes | Yes (with embedding) | Via embeddings |

Focus on which approaches naturally support borrowing strength across regions
and which require workarounds.

#### 5. Interaction effects explanation (CLIM-480)

Write a conceptual section explaining:

- How location-specific effects can be achieved through interaction terms
  (e.g. region x climate variable)
- The difference between regression models (which need explicit interaction terms)
  and flexible ML models (which can learn interactions implicitly)
- Practical examples showing when explicit interactions are needed vs not

### Open Questions

- What example dataset to use -- `laos_subset.csv` has multiple regions and is
  already used in the evaluation walkthrough.
- What underlying model should `BasicEstimator` use internally -- plain
  scikit-learn `LinearRegression`, or something with built-in uncertainty
  (e.g. Bayesian linear regression)?
- How to handle the noise model in `BasicEstimator` for generating `Samples` --
  Poisson (like `NaiveEstimator`), normal residuals, or negative binomial?

### Dependencies

- The evaluation walkthrough (CLIM-474, merged) provides the pattern for executable
  doc pages
- `NaiveEstimator` provides the structural pattern for `BasicEstimator`
- External tutorial repos on GitHub provide reference implementations
- `learn_modelling.md` provides the conceptual text that walkthroughs should complement
