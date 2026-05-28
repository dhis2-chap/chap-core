# explain-lime Command Reference

The `explain-lime` command generates a feature-by-feature importance
weighting for a single model prediction using LIME (Local Interpretable
Model-agnostic Explanations). It tells you which input features most
influenced the forecast for a specific location and horizon.

## Synopsis

```console
chap explain-lime --model-name <MODEL> --dataset-csv <CSV_FILE> --location <ORGUNIT> [OPTIONS]
```

## Description

LIME explains a prediction by:

1. Segmenting the historical features into time-window chunks.
2. Generating a few hundred *perturbations* of the original input that "turn off" different feature segments.
3. Running each perturbation through the original model.
4. Fitting a simple linear *surrogate* to the (perturbation, prediction) pairs, weighted by similarity to the original input.
5. Reading the surrogate's linear coefficients as the explanation: positive = pushed the forecast up; negative = pushed it down.

The full pipeline is segmenter → sampler → predict → weighter → surrogate; each stage is pluggable via `--lime-params.*` flags described below.

## Required Parameters

| Parameter | Description |
|-----------|-------------|
| `--model-name` | Model identifier. Local directory path under `runs/`, GitHub URL, or chapkit service URL. The directory must contain both an `MLproject` file and a trained `model` file — produce one with `chap eval` first. |
| `--dataset-csv` | Path to the CSV file containing the dataset to explain over. |
| `--location` | Name of the organisation unit whose prediction will be explained. Must exist in the dataset. |

## Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--horizon` | Number of future time steps to explain | 1 |
| `--lime-params.granularity` | Number of segments per feature | 10 |
| `--lime-params.num-perturbations` | Perturbations per run; higher = more stable but slower | 300 |
| `--lime-params.seed` | RNG seed for deterministic output | None |
| `--lime-params.last-n` | If set, only use the last N time steps of historical data | None |
| `--lime-params.timed` | Print per-stage timing | false |
| `--lime-params.adaptive` | Use `explain_adaptive` (Bayesian acquisition) instead of plain LIME | false |
| `--lime-params.with-metrics` | Also compute the `eLoss` faithfulness metric and log it | false |

## Choosing `--lime-params.num-perturbations`

LIME fits a linear surrogate over the interpretable features (roughly
`num_features × granularity` plus the future steps — typically 40–50 here),
so the perturbation count needs to comfortably exceed that for a stable fit.
A few times the feature count is the usual rule of thumb, which is why the
default is **300**. Lower counts give a faster but noisier explanation;
higher counts smooth the coefficients with diminishing returns.

The cost matters because each perturbation is one `model.predict` call. For
an **in-process** model that's cheap, so 300–1000 is comfortable. For an
**external** model (each predict is a subprocess/container) 300 calls can take
minutes, so:

- Prefer **`--lime-params.adaptive`** — it spends a smaller budget on the most
  informative perturbations (Bayesian acquisition), getting a comparable
  explanation with fewer model calls.
- Or lower `--lime-params.num-perturbations` toward ~100–150 (about 2–3× the
  feature count — the practical floor) and accept more variance.

**Verify you have enough**: re-run with two or three different
`--lime-params.seed` values. If the top features and the signs of their
coefficients are stable across seeds, the count is adequate; if they jump
around, raise it.

> Note: the small `--lime-params.num-perturbations` values used in the
> examples on this page are for a fast demo. Real explanations want the
> default (~300), or adaptive mode for slow models.

## Pipeline Components

LIME's four pluggable stages are selected by name. Defaults work fine for most cases; swap when investigating an unfaithful explanation.

### Segmenter — `--lime-params.segmenter-name`

How the time series is chunked. Determines what "a feature" means in the explanation.

| Name | Strategy |
|------|----------|
| `uniform` *(default)* | Equal-length chunks |
| `exponential` | Chunk sizes grow exponentially from the most recent end |
| `matrix_slope` | Matrix-profile boundaries at largest slope changes |
| `matrix_diff` | Matrix-profile boundaries sorted by largest slope changes |
| `matrix_bins` | Matrix-profile values binned into quantiles |
| `sax` | Symbolic Aggregate Approximation: segment by symbol runs |
| `nn` | Nearest-neighbour segmentation |

### Sampler — `--lime-params.sampler-name`

How a "turned off" segment is filled. Setting it to zero is rarely the right choice for time-series data because zero is semantically loaded (zero rainfall ≠ no rainfall signal).

| Name | Replacement |
|------|-------------|
| `background` *(default)* | Random draw from the dataset's typical values |
| `linear` | Linear interpolation between segment boundaries |
| `constant` | All zeros |
| `local_mean` | Mean of the segment itself, repeated |
| `global_mean` | Mean of the whole feature series, repeated |
| `random` | Uniform random draws from the feature's min/max |
| `fourier` | Short-time FFT replacement preserving dominant frequencies |

### Weighter — `--lime-params.weighter-name`

How each perturbation is weighted when fitting the surrogate.

| Name | Distance |
|------|----------|
| `pairwise` *(default)* | Euclidean on the mask vectors, RBF kernel |
| `dtw` | Dynamic Time Warping between perturbed and original sequences |

### Surrogate — `--lime-params.surrogate-name`

The interpretable model trained on perturbations.

| Name | Model |
|------|-------|
| `ridge` *(default)* | L2-regularised linear regression. Fast and stable. |
| `bayesian` / `blr` / `bayesian_linear` | Bayesian linear regression. Slower but produces uncertainty, used by adaptive mode. |

## Output Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--save` / `--no-save` | Write a Markdown explanation file under `runs/explainability/` | true |
| `--model-configuration-yaml` | Path to YAML with model-specific configuration | None |

## Run Configuration

Same as `chap eval`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--run-config.is-chapkit-model` | Set when explaining a chapkit REST API model | false |
| `--run-config.ignore-environment` | Skip automatic environment setup | false |
| `--run-config.debug` | Enable verbose debug logging | false |
| `--run-config.log-file` | Path to write log output | None |

## Example

```console
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

This runs LIME against an already-trained model on the dataset that was used to train it. With `--lime-params.with-metrics`, the `eLoss` faithfulness metric is computed and logged at the end.

## Output Format

The command logs to stdout. Key lines, in order:

1. **`Processing prediction chunk N/M`** — perturbations are run through the model in chunks of 10.
2. **Optional warnings** from the surrogate-input cleanup step:
   - `N/M perturbed predictions were negative; clipping to 0 before log1p` — the model produced negative values on some perturbations. Disease counts are non-negative, so negatives are the model going out of distribution. Clipping is a defensive fix; many clipped values means the explanation may not be trustworthy.
   - `N/M perturbations produced non-finite predictions; dropping them from the surrogate fit` — NaN or inf from the model. Rows are removed from the surrogate fit; a high drop rate is similarly suspect.
3. **`Surrogate weighted R2=… effective N=…, p=…`** — quality of the surrogate fit. R² measures how well the linear surrogate explains the model's perturbed responses; `n_eff` is the effective sample size after distance weighting.
4. **`Coefficients:`** section — one line per (feature × lag) and per (feature × future-step), sorted by `|coefficient|` descending. Positive = the feature pushed the forecast up; negative = pushed it down.
5. **`Faithfulness metrics:`** (only with `--lime-params.with-metrics`) — `r2`, `n_eff`, `delta_eloss`, `auc_top_k`, `auc_bottom_k`. See below.

## Interpreting `eLoss`

When `--lime-params.with-metrics` is passed, the CLI computes a *faithfulness* metric:

- The explanation's features are sorted by `|coefficient|` from most to least important.
- For each `k` in deciles, the top-k most important features are perturbed; the deviation `|y_perturbed − y_orig|` is measured. The same is done for the bottom-k least important features.
- `delta_eloss = AUC(top-k curve) − AUC(bottom-k curve)`.

Interpretation:

- **Strongly positive `delta_eloss`** — perturbing the features the explanation flagged as important moves the model more than perturbing the ones it flagged as unimportant. The explanation is faithful to the model.
- **Near zero or negative** — the explanation's importance ranking doesn't predict where the model is actually sensitive. Either the model is degenerate on perturbed inputs (see the warnings in step 2 above), or the LIME configuration needs adjustment (try a different `sampler-name` or fewer / longer segments).

## Prerequisites

`explain-lime` **does not train a model** — it is predict-only. It loads an *already-trained* model and probes it by running its `predict` step on perturbed inputs. So you must train the model first and point `--model-name` at the resulting run directory.

That directory must contain:

- an `MLproject` file, and
- a trained artifact **literally named `model`** — the file the model's own predict script loads. chap-core never deserialises the model itself; it just invokes the model's predict entry point against this file. The name is fixed, not configurable: chap-core always passes the literal `model` as the `{model}` parameter of the MLproject `train`/`predict` commands, so the trained file in the run directory is always exactly `model`.

A bare GitHub URL (e.g. `https://github.com/dhis2-chap/...`) resolves to the model *template* (code only, no trained `model` file), so it will **not** work as `--model-name` here.

Train first with `chap eval` against the dataset, then pass the produced run directory:

```console
chap eval \
    --model-name <github-or-local-model> \
    --dataset-csv <CSV> \
    --output-file <OUT.nc>
```

Then pass the resulting `runs/<name>/<timestamp>_<hash>/` directory as `--model-name` here.

!!! note "Verify the `model` file exists"
    Not every model leaves a reusable `model` artifact in the run directory after a backtest. Before running `explain-lime`, check that the run directory actually contains a file named `model` (`ls runs/<name>/<timestamp>_<hash>/`). If it doesn't, `explain-lime` fails when the model's predict step tries to load it (e.g. `cannot open compressed file 'model'`).

## See Also

- The original [LIME paper (Ribeiro et al., 2016)](https://arxiv.org/abs/1602.04938).
- [Model Explainability](../external_models/model_explainability.md) — a separate, model-author-side feature where external models save their own coefficient files during prediction. Different concept from this command.
