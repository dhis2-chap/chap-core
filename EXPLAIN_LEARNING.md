# EXPLAIN_LEARNING.md — slow walkthrough for tomorrow

> **NOT FOR MERGE.** Companion to `EXPLAIN.md`. Written so you can sit down
> tomorrow morning, read this top-to-bottom with a coffee, and end up
> understanding what LIME is, what your PR did, and what the output of
> `chap explain-lime` actually means. Delete before merge.

Where `EXPLAIN.md` is a *reference* (with tables, diagrams, file maps),
this is a *narrative*: a single end-to-end story using the Laos dataset
that's already in the repo. Real numbers, real outputs, real commands.

If you only have 15 minutes: read parts 1–3.
If you have an hour: do parts 1–6.
If you want to know it cold: do everything.

---

# Part 1 — The big question

## Who's asking, and what for

Imagine you're a public-health official in Bokeo, Laos. CHAP has a
trained model that, every month, looks at the last few months of
rainfall + temperature + recorded disease cases, and forecasts how many
malaria cases you should expect next month. Last month it predicted
"15 cases." Reality was 4. The month before, it predicted "12" and you
got 22. You'd like to know:

- *Why does the model think next month will be high/low?*
- *Is it reacting to last week's rain, or to a heat wave six months ago,
  or to the long-term trend in cases?*
- *Should I trust this forecast enough to deploy spraying teams to
  particular villages?*

The model itself is a black box. Internally it might be a neural
network with thousands of weights, or an XGBoost ensemble of hundreds
of trees. There is no human-readable answer to "why."

**Explainability is the bridge from prediction to "because."**

LIME (Local Interpretable Model-agnostic Explanations) is one way to
build that bridge: you build a *tiny* interpretable model that mimics
the big opaque model around a single prediction, then read the tiny
model's coefficients as "X mattered this much, Y mattered that much,
Z barely mattered."

It is **local** — only valid around the one prediction you're
explaining. It is **model-agnostic** — it doesn't need to look inside
the black box, only call its `.predict(...)` method.

That's what `chap_core.explainability` implements, and that's what
`chap explain-lime` runs from the CLI.

## What this PR did and didn't

It did:

1. Made the lint pass cleanly on the explainability subpackage — the
   `#262` carve-outs hid real type errors and a broken import.
2. Shipped the **eLoss faithfulness metric** that #262 referenced but
   never delivered (the `chap_core.explainability.testing.metrics`
   module was missing on master).
3. Fixed a **`log1p` NaN crash** that made `chap explain-lime`
   unusable against any real model. The pipeline ran fine up to the
   surrogate-fit step, then crashed with `ValueError: Input y contains
   NaN.` This PR clips negative model outputs and drops non-finite
   rows before the surrogate fit.
4. Added 68 unit + integration tests where there were zero. The
   integration test mocks a model so the whole pipeline gets exercised
   in CI without needing a trained `runs/` directory.
5. Added a `--lime-params.with-metrics` CLI flag so operators can see
   the eLoss number from the terminal.
6. Wrote a permanent CLI reference (`docs/chap-cli/explain-lime-reference.md`)
   modelled on the existing `eval-reference.md`.

It did not:

- Change what LIME *is* or what the pipeline *does* — those came from
  PR #262.
- Train any new models or change any forecasting behaviour. The black
  boxes still produce the same predictions; only the explanations are
  now visible.
- Implement explainability for non-LIME methods (e.g. SHAP). LIME is
  the only algorithm here.

---

# Part 2 — The Laos data (with real numbers)

There's a tiny demo dataset checked into the repo at:

```
runs/minimalist_example_uv/2026-05-26_20-43-33_d8514acc/training_data.csv
```

Three provinces in Laos (Bokeo, Savannakhet, Vientiane[prefecture]).
34 months each, January 2010 through October 2012. Five real columns:
`time_period`, `rainfall`, `mean_temperature`, `disease_cases`, `population`.

First three rows for Bokeo:

```
time_period  rainfall  mean_temperature  disease_cases  population
2010-01        37.97             20.04            1.0     75049.56
2010-02         8.53             22.22            1.0     75049.56
2010-03        23.59             24.59            2.0     75049.56
```

Last three:

```
2012-08       362.18             23.70           17.0     78358.92
2012-09       242.42             23.69           26.0     78358.92
2012-10       107.07             22.96           28.0     78358.92
```

Summary for Bokeo:

| Stat | rainfall | mean_temperature | disease_cases |
|---|---|---|---|
| min | 5.97 mm | 18.16 °C | 0 |
| median | 106.72 mm | 23.51 °C | 4.5 |
| max | 512.74 mm | 27.52 °C | 28 |

Three things to notice:

1. **Monthly granularity.** One row per month per province. So "lag 1"
   means "one month ago", "lag 5" means "five months ago".
2. **Strong rainfall seasonality.** Range 5.97 to 512.74 mm — the wet
   season really is two orders of magnitude wetter than the dry
   season. This matters for LIME because perturbing rainfall in the
   wet season vs the dry season is a very different thing.
3. **Sparse disease counts.** Median 4.5 cases per month, max 28. So
   even small perturbations to the input can dominate the prediction.

The geojson polygons for these three locations live at
`example_data/laos_subset.geojson` (this trained run wasn't
GeoJSON-paired, but conceptually it could have been).

---

# Part 3 — What the model actually does (the toy version)

The trained model in `runs/minimalist_example_uv/.../model` is *very
simple*. Open `runs/minimalist_example_uv/.../train.py` and you'll see:

```python
from sklearn.linear_model import LinearRegression
df = pd.read_csv(train_data_path)
features = df[["rainfall", "mean_temperature"]].fillna(0)
target = df["disease_cases"].fillna(0)
model = LinearRegression()
model.fit(features, target)
```

That's it. A two-feature linear regression — disease_cases as a function
of `rainfall` and `mean_temperature`, fit globally across all three
provinces and all 34 months. No lags, no interactions, no per-location
intercepts, no population scaling.

Three things to know about it:

1. **It can produce negative case counts.** Linear regression doesn't
   constrain outputs to ≥ 0. When you feed it weird perturbed inputs
   (e.g. negative rainfall from a noisy sampler), it happily
   extrapolates into negative-cases territory.
2. **It's a stand-in for "any model with `.predict()`."** LIME is
   model-agnostic; the same explanation machinery works against
   xgboost, an INLA fit, a chapkit-hosted PyTorch model, anything.
   The toy linear regression is the *least* interesting thing the
   pipeline will ever explain, but it's the easiest to ship a CI
   fixture for.
3. **Its real coefficients are not what LIME's coefficients should
   recover.** The model has two coefficients (one for rainfall, one
   for temperature). LIME, when given segmented historical data and
   future climate, will produce dozens of coefficients (one per
   feature-lag pair). LIME's output is a *local* approximation of how
   the model behaves *around the input you're explaining*, not a
   summary of the model's internals.

`runs/minimalist_example_uv/.../predict.py` is similarly compact:

```python
model = joblib.load(model_path)
future_df = pd.read_csv(future_data_path)
features = future_df[["rainfall", "mean_temperature"]].fillna(0)
predictions = model.predict(features)
output_df = future_df[["time_period", "location"]].copy()
output_df["sample_0"] = predictions
output_df.to_csv(out_file_path, index=False)
```

So calling `model.predict(historic_data, future_data)` produces a CSV
with `time_period`, `location`, and one `sample_0` column. The "samples"
naming is because in general a model can return multiple Monte Carlo
samples per prediction; this toy model only returns one.

---

# Part 4 — The journey: data → backtest → forecast → explanation

There are three things you can do with a CHAP model. Make sure these
feel distinct in your head before going further:

| Operation | Inputs | Output | What it tells you |
|---|---|---|---|
| **Backtest** (`chap eval`) | full dataset, model | NetCDF of per-split forecasts + truth | *How well does this model forecast in general?* |
| **Forecast** (one-shot via Python or `chap_core.predictor`) | full historical data, model, future horizon | a single forecast for the next N months | *What does this model say next month looks like?* |
| **Explain** (`chap explain-lime`) | a forecast already produced, model, dataset | feature-by-feature importance weights | *Why did the model say that?* |

LIME runs *after* the model has made its prediction. It re-runs the
model many more times on perturbed inputs to *probe* it, but it doesn't
need to retrain.

Sequence in practice:

```
       data CSV
          │
          ▼
   ┌───────────────────────┐    ┌───────────────────────┐
   │  chap eval            │    │  chap explain-lime    │
   │  rolling backtest     │    │  LIME on one          │
   │  produces .nc + runs/ │    │  prediction           │
   └───────────────────────┘    └───────────────────────┘
          │                              ▲
          ▼                              │
   trained runs/<x>/<ts>/                │  needs a trained
   (model + MLproject)                   │  runs/ directory
          │                              │  produced by `chap eval`
          └──────────────────────────────┘
```

So `chap eval` is the prerequisite for `chap explain-lime`. The
existing CLI doesn't have a separate "forecast" command — backtests
implicitly produce forecasts internally as they roll over the data.

---

# Part 5 — A LIME explanation, walked through step by step

Let's say you want to explain the forecast the toy model produced for
**Bokeo in November 2012, looking 3 months ahead**. That's exactly the
invocation we've been using:

```bash
chap explain-lime \
    --model-name runs/minimalist_example_uv/2026-05-26_20-43-33_d8514acc \
    --dataset-csv runs/minimalist_example_uv/2026-05-26_20-43-33_d8514acc/training_data.csv \
    --location Bokeo \
    --horizon 3 \
    --lime-params.num-perturbations 30 \
    --lime-params.seed 42 \
    --lime-params.with-metrics \
    --no-save
```

What follows is what LIME does internally, in seven stages. Read each
stage with the original ASCII pipeline (EXPLAIN.md §3) open beside it.

## Stage 1 — Get the original input vector x₀

LIME first asks: *what's the input you want to explain?* It's a pair
of dataframes:

- **hist_df**: historical data for Bokeo (one row per month, 34 months
  from 2010-01 to 2012-10). Columns: `rainfall`, `mean_temperature`,
  `disease_cases`, `population`.
- **fut_df**: future "weather" — what climate covariates will look like
  for the 3 months we want to forecast (2012-11, 2012-12, 2013-01).
  This is *synthesized* by chap-core's built-in `MonthlyClimatePredictor`
  (a regression on month-of-year features) because the real future
  weather isn't known when we run the explanation.

In code (`chap_core/explainability/lime.py:932-960`):

```python
climate_predictor = get_climate_predictor(climate_data)
full_future_weather = climate_predictor.predict(prediction_range)
dataset_loc = dataset.filter_locations([location])
future_weather = full_future_weather.filter_locations([location])
hist_df = dataset_loc.to_pandas().sort_values("time_period").reset_index(drop=True)
future_df = future_weather.to_pandas().sort_values("time_period").reset_index(drop=True)
```

For us, hist_df has 34 rows × 4 features and fut_df has 3 rows × 2
features (rainfall and mean_temperature). The "vector to explain" is
this pair (hist_df, fut_df).

## Stage 2 — Segment historical data

Now the trick that makes LIME work on time series: instead of treating
each of the 34 historical months as 34 separate features (which would
require thousands of perturbations to cover), we **segment** each
historical column into *granularity* chunks. Default `granularity=10`.

For Bokeo's rainfall column (34 values), the default `uniform`
segmenter produces 10 segments. The math: 34 // 10 = 3 months per
segment, with the last segment getting the remainder (3*9 = 27, so the
last segment is months 28–34, i.e. 7 months). Concretely:

| Lag | Rows | Time periods | Rainfall values (mm) |
|---|---|---|---|
| lag_9 | 0–3 | 2010-01 … 2010-03 | 37.97, 8.53, 23.59 |
| lag_8 | 3–6 | 2010-04 … 2010-06 | 101.18, ?, ? |
| lag_7 | 6–9 | … | … |
| … | … | … | … |
| lag_0 | 27–34 | 2012-04 … 2012-10 | …, 362.18, 242.42, 107.07 |

(`lag_0` is the *most recent* segment; higher lag = older.)

Same segmentation happens for `mean_temperature`, `disease_cases`,
`population`. After this stage you have a structure called `x0` (the
"original vector") that looks like a nested dict:

```python
x0 = {
  "rainfall":     {0: [..., 362.18, 242.42, 107.07], 1: [...], ..., 9: [37.97, 8.53, 23.59]},
  "mean_temperature": {0: [...], ..., 9: [20.04, 22.22, 24.59]},
  "disease_cases": {0: [...], ..., 9: [1.0, 1.0, 2.0]},
  "population":   {0: [...], ..., 9: [75049.56, 75049.56, 75049.56]},
}
```

Plus a parallel structure `feat_indices` mapping `(feature, lag)` to
`(start_row, end_row)` so we can later restore perturbed values into
the dataframe.

Code: `chap_core/explainability/lime.py:101-146` (`build_original_vector`)
+ `chap_core/explainability/segment.py` (the segmenter classes).

Each historical feature × each lag becomes one "interpretable feature."
For 4 historical columns × 10 lags = 40 historical features. Plus
3 future features × 3 horizons = 9 future features. Plus the
`population` feature is *constant* over time so it gets one slot per
horizon instead of segmented. Total: **49 features** in the
`feature_map`. That's why you see `p=49` in the surrogate's R² log line.

## Stage 3 — Build perturbation masks

Now we generate ~300 (here: 30) random 0/1 vectors of length 49. Each
vector represents one "perturbed reality": where 1 means "keep the
original segment value" and 0 means "turn this segment OFF."

For example, mask might look like:

```
[1, 0, 1, 1, 1, 0, 1, 1, 1, 1,   # rainfall lags 9..0
 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,   # mean_temperature lags 9..0
 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,   # disease_cases lags 9..0
 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,   # population lags 9..0
 1, 1, 0,                        # rainfall_fut_1, _fut_2, _fut_3
 1, 0, 1,                        # mean_temperature_fut_1, _fut_2, _fut_3
 1, 1, 1]                        # population_fut_1, _fut_2, _fut_3
```

This particular mask says: "show me what the model predicts if I keep
*most* of the historical and future inputs as-is, but pretend I don't
know rainfall_lag_8, mean_temperature_lag_5, disease_cases_lag_3,
population_lag_2, rainfall_fut_3, and mean_temperature_fut_2."

Code: `chap_core/explainability/lime.py:149-184` (`create_masks`).

## Stage 4 — Materialise the perturbations (the sampler step)

For every "0" in every mask, we need a *replacement value* — what
should go in place of the segment we're "turning off"? This is where
the `sampler_name` parameter matters. The default `background` sampler
picks values from the dataset's typical distribution; `local_mean`
uses the segment's own mean; `linear` interpolates between segment
boundaries; etc.

For the mask above, the sampler would produce something like:

```
Original rainfall lag_8 (rows 3..6): [101.18, 78.45, 156.20]
Perturbed (background sampler):    [180.55, 95.10, 122.30]   # random draws from typical rainfall
```

The end result of this stage is a *list of perturbed (hist_df, fut_df)
pairs*. One pair per mask.

Code: `chap_core/explainability/lime.py:187-253` (`perturb_vectors`)
+ `chap_core/explainability/perturb.py` (the sampler classes).

## Stage 5 — Run the black box on every perturbation

Now LIME calls `model.predict(perturbed_hist_df, perturbed_fut_df)`
for each perturbation. With 30 perturbations and horizon=3, that's
30 separate forecasts. To make this efficient, the pipeline groups
perturbations into "pseudo-locations" (`pb_0`, `pb_1`, …) and calls
the model in chunks of 10:

```
Processing prediction chunk 1 (10 perturbations)...
Processing prediction chunk 2 (10 perturbations)...
Processing prediction chunk 3 (10 perturbations)...
```

Each chunk runs the model once, with a dataframe whose `location`
column is `pb_0` through `pb_9`. The model treats these as 10 distinct
locations and produces 10 sets of forecasts.

What comes back is an array of 30 predictions (one per perturbation,
averaged across the horizon). For the toy linear-regression model on
the Bokeo data, every single one of these 30 predictions came out
**negative** in my test run, which is the warning you saw:

```
WARNING 30/30 perturbed predictions were negative; clipping to 0 before log1p
```

Why negative? Because:

- The model was trained on `rainfall` ranging 5.97 → 512.74 and
  `mean_temperature` ranging 18.16 → 27.52.
- The `background` sampler picks replacement values *from those same
  ranges*, but in combinations the model never saw during training.
- A linear regression with negative coefficient on temperature
  (because cooler months in this dataset happen to have fewer cases)
  will extrapolate negatively when the perturbed temperature is at the
  high end of its range simultaneously with a very low rainfall.

This is the famous "model goes out of distribution under perturbation"
problem in LIME. Real models trained with non-negative output
constraints (Poisson regression, ReLU final layer, etc.) wouldn't hit
this.

Code: `chap_core/explainability/lime.py:545-664` (`produce_lime_dataset`).

## Stage 6 — Weight, log-transform, and fit the surrogate

LIME doesn't just train a linear regression on (mask, prediction)
pairs. It does three more things first:

**6a. Distance weighting.** Each perturbation is closer or farther
from the original. A perturbation that turned off only 1 segment is
"close"; one that turned off 30 is "far." LIME weights each pair by
this distance — closer perturbations matter more for explaining the
*local* behaviour. The default `pairwise` weighter computes Euclidean
distance in mask space and applies an RBF kernel.

**6b. Log-transform.** Disease case counts have a long right tail
(median 4.5, max 28). LIME applies `log1p(y) = log(1+y)` to compress
that. **This is also where the new helper kicks in**: any `y < 0`
gets clipped to 0 first (so `log1p` doesn't return NaN), and any row
where the model returned NaN/inf gets dropped from the surrogate fit.

**6c. Surrogate fit.** Linear regression (Ridge by default) on the
(mask, log_prediction, weight) tuples. The coefficients of this linear
regression are the explanation.

For our run:

```
Surrogate weighted R2=1.000, effective N=30.0, p=49
```

`R²=1.0` looks great, but here it's a tell-tale sign of degenerate
input: after clipping all 30 predictions to 0, the log-transformed
target is constant 0, and any linear model fits a constant target
trivially with R²=1.0 and all coefficients = 0. This is *informative*:
it tells you the explanation is meaningless because the model can't
distinguish the perturbations.

Code: `chap_core/explainability/lime.py:412-460` (the
`_log_transform_for_surrogate` helper this PR added) and
`chap_core/explainability/lime.py:1080-1085` (the surrogate fit
call site).

## Stage 7 — Sort and present the coefficients

The surrogate's `.coef_` array, paired with the feature names, gets
sorted by absolute magnitude descending. For our toy run all
coefficients are zero, so the sort is arbitrary — but for a real model
on a real dataset, you'd see the most-influential features at the top
with positive (pushed forecast up) or negative (pushed forecast down)
coefficients.

Code: `chap_core/explainability/surrogate.py:18-20`
(`SurrogateResult.as_sorted`).

---

# Part 6 — The eLoss faithfulness metric

The seven-stage pipeline above produces an explanation. But how do you
know the explanation isn't lying?

A reasonable test: if the explanation says feature X is the most
important, then *perturbing X should move the model's prediction more
than perturbing some feature Y that the explanation says doesn't
matter*. If perturbing the "important" features barely changes the
model's output, the explanation isn't capturing what's actually going
on inside the model — it's bullshit.

That's the eLoss metric, from Nguyen / Le Nguyen / Ifrim:

1. Sort the explanation's features by `|coefficient|`.
2. For each `k` in [10%, 20%, …, 100%]:
   - Build a perturbation that turns off only the **top-k** most
     important features. Measure how much the prediction changes:
     `|y_perturbed − y_orig|`.
   - Same for the **bottom-k** least important features.
3. You now have two curves of (k, deviation). The top-k curve should
   rise fast (perturbing important features moves the model a lot).
   The bottom-k curve should rise slowly.
4. Take trapezoidal AUC of each, then **`delta_eLoss = AUC(top-k) −
   AUC(bottom-k)`**. Higher = explanation is more faithful.

For our toy run:

```
EVALUATION: Delta eLoss = -13985.18 (Top-k AUC: 941.47, Bottom-k AUC: 14926.65)
Faithfulness metrics:
                r2 = +1.0000
             n_eff = +29.99
       delta_eloss = -13985.18
         auc_top_k = +941.47
      auc_bottom_k = +14926.65
```

`delta_eloss` is strongly *negative*. Reading: the explanation
(all-zero coefficients, arbitrary tie-breaking ordering) says the
"top" features are random ones, but perturbing the "bottom" features
moves the model 16× more than perturbing the "top." The explanation
is anti-correlated with what actually drives the model. Which is
correct! The explanation captured nothing.

Implementation: `chap_core/explainability/testing/metrics.py` (this
PR's most user-visible addition).

---

# Part 7 — The `log1p` fix in plain English

The crash on master was specifically here:

```python
# explain() in chap_core/explainability/lime.py
z = np.log1p(y)                       # ← y has some values < -1
surrogate_model.fit(X, z, weights)    # ← sklearn sees NaN in z, refuses
```

`np.log1p(y) = log(1 + y)`. For `y = -2`, that's `log(-1)`, which is NaN
in real arithmetic. sklearn's `Ridge.fit()` then raises `ValueError:
Input y contains NaN.` and the whole pipeline crashes.

Why does the model produce `y ≤ -1` in the first place? Two reasons:

1. **Wrong model class for the target.** The toy model is unconstrained
   linear regression on a non-negative target. Real disease forecasting
   would use Poisson regression or negative binomial regression
   (always-positive output by construction) or a model with a softplus
   final layer.
2. **Out-of-distribution perturbations.** LIME's perturbations move
   the input to combinations the model never saw at training time.
   Even a well-trained model can extrapolate weirdly when asked to
   predict at inputs it's never encountered.

The fix this PR introduced is the helper
`_log_transform_for_surrogate(X, y, weights)`:

```python
y = np.asarray(y, dtype=float)
n = y.shape[0]

# 1. Clip negatives to 0 (log1p(0) = 0 is safe).
neg_mask = y < 0
if neg_mask.any():
    logger.warning("%d/%d perturbed predictions were negative; clipping...", ...)
    y = np.clip(y, 0.0, None)

z = np.log1p(y)

# 2. Drop rows where z is still non-finite (model returned NaN/inf directly).
finite_mask = np.isfinite(z)
if not finite_mask.all():
    logger.warning("%d/%d perturbations produced non-finite predictions; dropping...", ...)
    if not finite_mask.any():
        raise ValueError("All perturbed predictions were non-finite; cannot fit surrogate")
    X = X[finite_mask]
    z = z[finite_mask]
    weights = weights[finite_mask]

return X, z, weights
```

In English: "If the model returned a value below zero, clip it to zero
(disease counts can't be negative). If the model returned NaN or
infinity, throw that row out entirely. Warn loudly so the operator
knows something is off. Crash only if *every* perturbation failed."

This makes degenerate cases visible (the warnings tell you exactly how
many perturbations went OOD) instead of failing with an opaque sklearn
traceback.

## Follow-up: don't forget to use the filtered X downstream

There's a subtle gotcha that the first attempt of this fix missed. The
helper returns the *filtered* `(X_fit, z, weights)` — rows where the
model produced NaN/inf are dropped. But the original `explain()` /
`explain_adaptive()` then computed R² with `surrogate_model.predict(X)`
on the **unfiltered** `X`. As soon as *some* perturbations went
non-finite (but not all), `r2_score(z, y_pred, sample_weight=weights)`
got mismatched lengths:

```
ValueError: Found input variables with inconsistent numbers of
samples: [10, 20, 10]
```

A reviewer caught it on the PR. The fix is one-line: use `X_fit`
instead of `X` in the R² call (and in the `p=` / `n=` log line too,
so the diagnostic matches what the surrogate actually saw). The
all-NaN case raises before reaching the surrogate; the all-finite
case has `X_fit == X`; only the partial-NaN case is affected. There's
now a regression test
(`test_mixed_finite_and_non_finite_predictions_complete_without_length_mismatch`)
that reproduces the exact mismatch without the fix and passes with it.

---

# Part 8 — Hands-on: what to run tomorrow

These are short — no exercise should take more than a few minutes.

## Exercise 1: see the data

```bash
uv run python <<'PY'
import pandas as pd
df = pd.read_csv("runs/minimalist_example_uv/2026-05-26_20-43-33_d8514acc/training_data.csv")
print("locations:", df['location'].unique().tolist())
print("months:   ", df['time_period'].nunique(), "from",
      df['time_period'].min(), "to", df['time_period'].max())
print("\nBokeo rainfall by year-month:")
bokeo = df[df['location']=='Bokeo'][['time_period','rainfall','disease_cases']]
print(bokeo.to_string(index=False))
PY
```

You'll see all 34 months with their rainfall and case counts. Notice
the strong wet-season spike in 2010-08 and 2011-08 (>400mm rainfall)
and the corresponding small bump in cases.

## Exercise 2: run the unit tests

```bash
uv run pytest tests/explainability/ -v
```

Expect 68 passing in ~3 seconds. Skim a few test names — they're
named after what they verify. The integration tests in
`test_explain_integration.py` are the ones that exercise the full
pipeline against a mock model; the rest are unit tests on individual
components.

## Exercise 3: run the eLoss math demo

```bash
uv run python <<'PY'
import numpy as np
from chap_core.explainability import lime as lime_module
from chap_core.explainability.testing.metrics import eLoss

feature_names = [f"f{i}" for i in range(10)]
# Truth: only the first 5 features actually matter
importance_truth = np.array([1.0]*5 + [0.0]*5)

# Fake the perturb/predict steps so we can isolate just the math
def fake_perturb(*args, **kw): return args[5], args[5]
def fake_produce(*args, **kw):
    perturbations = args[3]
    ys = [float(np.sum(importance_truth * (m == 0))) for m in perturbations]
    return None, np.asarray(ys), None, None

lime_module.perturb_vectors = fake_perturb
lime_module.produce_lime_dataset = fake_produce

common = dict(model=None, original_vector={}, feature_map=[], sampler=None,
              hist_df=None, fut_df=None, features_hist=[], features_fut=[],
              horizon=1, location="loc", hist_type=None, fut_type=None,
              feat_indices={}, y_orig=0.0, full_dataset=None,
              full_future_weather=None)

# Faithful ranking: lists features in true importance order
faithful = [(f"f{i}", 10.0 - i) for i in range(10)]
delta_f, _, _ = eLoss(**common, feature_names=feature_names, sorted_explanation=faithful)

# Anti-faithful: lists features in REVERSE order (5..9 marked most important)
anti = [(f"f{i}", 1.0 + i) for i in range(9, -1, -1)]
delta_a, _, _ = eLoss(**common, feature_names=feature_names, sorted_explanation=anti)

print(f"Faithful explanation:      delta_eloss = {delta_f:+.1f}")
print(f"Anti-faithful explanation: delta_eloss = {delta_a:+.1f}")
PY
```

You should see `+24.5` and `-24.5` — perfect mirror image. Faithful
explanation has positive delta_eloss; anti-faithful has negative.

## Exercise 4: run the CLI against the trained model

```bash
uv run chap explain-lime \
    --model-name runs/minimalist_example_uv/2026-05-26_20-43-33_d8514acc \
    --dataset-csv runs/minimalist_example_uv/2026-05-26_20-43-33_d8514acc/training_data.csv \
    --location Bokeo \
    --horizon 3 \
    --lime-params.num-perturbations 30 \
    --lime-params.seed 42 \
    --lime-params.with-metrics \
    --no-save
```

Reading the output:
- 3 chunks of 10 perturbations = 30 total. Then the warning fires:
  "30/30 perturbed predictions were negative; clipping to 0 before
  log1p". That's the model going OOD on every single perturbation.
- The surrogate fits R²=1.0 trivially because z is constant 0.
- Every coefficient comes out 0.
- The eLoss block at the end says `delta_eloss ≈ -13985`, confirming
  the explanation is not faithful (because there is no explanation
  there to be faithful — the surrogate captured nothing).

## Exercise 5: try a different sampler

The default `background` sampler is the one going OOD here. Try
`local_mean`:

```bash
uv run chap explain-lime \
    --model-name runs/minimalist_example_uv/2026-05-26_20-43-33_d8514acc \
    --dataset-csv runs/minimalist_example_uv/2026-05-26_20-43-33_d8514acc/training_data.csv \
    --location Bokeo \
    --horizon 3 \
    --lime-params.num-perturbations 30 \
    --lime-params.sampler-name local_mean \
    --lime-params.seed 42 \
    --lime-params.with-metrics \
    --no-save
```

`local_mean` replaces each perturbed segment with the mean of that same
segment — much closer to the original input than the `background`
sampler, so the model stays in-distribution. You should see fewer (or
zero) perturbations getting clipped, real non-zero coefficients in the
listing, and a more meaningful eLoss number.

Compare: the meaning of the coefficients depends on the sampler! There
isn't *one* explanation for a prediction — there's one explanation per
(segmenter, sampler, weighter, surrogate) tuple. That's both LIME's
flexibility and its biggest critique in the literature: explanations
aren't unique.

## Exercise 6: look at the trained-run files

```bash
ls runs/minimalist_example_uv/2026-05-26_20-43-33_d8514acc/
```

You'll see:

| File | Purpose |
|---|---|
| `MLproject` | mlflow project descriptor |
| `model` | the joblib-pickled sklearn LinearRegression |
| `train.py` / `predict.py` | the toy model's training and prediction scripts |
| `python_env.yml` | environment spec |
| `historic_data_<date>.csv` | data passed to the model during prediction |
| `future_data_<date>.csv` | future weather passed to the model |
| `predictions_<date>.csv` | the model's output |
| `training_data.csv` | the dataset the model was trained on |

`historic_data_*.csv` and `future_data_*.csv` are interesting because
they show the format the model receives during a perturbation call.
Each row has a `location` column — when LIME batches perturbations,
that column contains `pb_0`, `pb_1`, … instead of "Bokeo."

## Exercise 7: what does the integration test do?

Read `tests/explainability/test_explain_integration.py` end to end —
it's only ~180 lines. Note especially:

- The `MockExternalModel` class — a deterministic stand-in for any
  trained model. The PR added this so the full LIME pipeline can be
  exercised in CI without a `runs/` directory.
- `_explain_with_defaults(...)` — wraps the `explain()` call with safe
  test defaults (no saving, no plotting, small num_perturbations).
- The `TestLog1pHelperEndToEnd` class — proves the fix this PR shipped
  works end-to-end against the real pipeline (not just the helper in
  isolation).

If you ever break something while editing the explainability module,
running just this file (`pytest tests/explainability/test_explain_integration.py
-v`) is the fastest way to find out.

---

# Part 9 — Glossary, expanded

(Same terms as EXPLAIN.md §8 but with more colour.)

- **x₀ ("x naught") / `original_vector`** — the unperturbed inputs you
  want to explain. In our case, the historical data for Bokeo plus
  3 months of synthesised future climate.

- **segment** — a chunk of consecutive time steps in a historical
  feature. With granularity=10 and 34 months of data, segments are
  3–7 months wide each.

- **lag** — segments numbered from most recent (lag_0) to oldest
  (lag_9). "rainfall_lag_3" means "the rainfall segment from a few
  months ago, depending on granularity and total history."

- **fut / future feature** — a feature whose values are *expected to
  be known* at prediction time. For chap-core that's typically just
  climate (rainfall_fut_1 = "predicted rainfall for the first
  forecast month").

- **flat mask** — a numpy array of 0s and 1s, one entry per
  interpretable feature (49 for our run). 1 means "keep original
  segment", 0 means "replace it with a sampler-generated value."

- **feature_map** — a list of `(name, parent_key, lag)` triples that
  tells you what each position in a flat mask corresponds to.
  Position 0 is `("rainfall_lag_9", "rainfall", 9)`, etc.

- **perturbation** — the result of applying one mask to x₀: a new
  (hist_df, fut_df) pair with some segments replaced. The pipeline
  generates `num_perturbations` of these.

- **surrogate** — a small interpretable model (Ridge or Bayesian
  linear regression) trained on the perturbation→prediction pairs.
  Its coefficients *are* the explanation.

- **R²** — how well the surrogate's predictions match the black box's
  predictions on the perturbations, weighted by distance. A *high*
  R² means the surrogate is a good local approximation. A *trivial*
  R² (like our 1.0 on a constant target) means the surrogate had
  nothing to learn.

- **n_eff** — effective sample size after distance weighting. Lower
  than `num_perturbations` because perturbations far from x₀ are
  down-weighted. In our run n_eff = 29.99 / 30 — the weighter barely
  down-weighted anything because the perturbations weren't very
  varied.

- **eLoss / delta_eloss** — the faithfulness metric. Compares how
  much the model's prediction changes when you perturb the
  explanation's "important" vs "unimportant" features. Positive =
  faithful; near zero or negative = misleading.

- **sampler going OOD** — the perturbation sampler picks values from
  the original dataset's distribution, but in combinations the model
  has never seen. Different samplers have different OOD behaviour;
  `local_mean` is the most conservative, `background` and `random`
  the most aggressive.

---

# Part 10 — Cheat sheet for the PR review

If someone asks you a question about this PR tomorrow, here are the
short answers:

| Question | One-line answer |
|---|---|
| "What was actually broken on master?" | Two things: the `chap_core.explainability.testing.metrics` import didn't resolve (eLoss path crashed with ImportError when `return_metrics=True`), and `np.log1p(y)` crashed sklearn when the model produced negative predictions on perturbed inputs. |
| "Why was nobody noticing?" | The two lint carve-outs in `pyproject.toml` (F403/F405 + mypy override) hid the broken import. The `log1p` crash only fired when actually running `chap explain-lime` against a trained model, which had no integration test. |
| "What does this PR actually change about LIME's algorithm?" | Nothing. The 7-stage pipeline (segment → mask → perturb → predict → weight → surrogate → sort) is unchanged. We added one defensive helper (`_log_transform_for_surrogate`) on the surrogate-fit side and an eLoss faithfulness metric on top. |
| "Why are all the coefficients zero in the demo?" | Because the toy linear regression model produces uniformly negative predictions on every perturbation (model goes OOD on the perturbed inputs). After clipping to 0, the log-transformed target is constant 0, so the surrogate fits a constant. Real models trained with non-negative output constraints (Poisson, etc.) won't hit this. |
| "Why are the deps now mandatory instead of optional?" | Every module in `chap_core.explainability` imports `stumpy`/`pyts`/`fastdtw` at module top-level — no try/except guards. So they were already de-facto required. The "optional" classification just made `uv sync --dev` (CI's default) fail to collect the new test suite. |
| "What's the next thing to fix in this area?" | Nothing urgent for explainability itself. Possible follow-ups listed in EXPLAIN.md §13: auto-render CLI reference via cyclopts's mkdocs plugin, lift the EXPLAIN.md architecture content into `docs/contributor/lime_pipeline.md`, add `--output-csv` for programmatic consumers. |

---

# Part 11 — Things worth reading after this

Inside the repo:

- `EXPLAIN.md` — same scope but reference-style instead of narrative.
- `docs/chap-cli/explain-lime-reference.md` — the permanent CLI
  reference page (this PR added).
- `chap_core/explainability/userguide.md` — the original author's
  user-facing intro. Predates this PR.

Outside the repo:

- The LIME paper (Ribeiro et al., 2016): https://arxiv.org/abs/1602.04938 —
  short, clear, easy read.
- Leander Skoglund's MSc thesis (the PDF at `~/Downloads/2026-leander.pdf`)
  — the chap-core adaptation. The eLoss algorithm is in chapter 5.
- For a critique of LIME's limitations:
  *"Why Should I Trust You?"* discussion in the SHAP paper (Lundberg
  & Lee, 2017) https://arxiv.org/abs/1705.07874 — SHAP is the
  next-generation alternative.

When you're ready to dig into any specific section, ping me and we go
slowly through it together.
