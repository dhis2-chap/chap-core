# Run many evaluations with a loop

When you compare models, tune a parameter, or test different backtest settings, you end
up running `chap eval` over and over on the **same dataset**. Doing that by hand is slow
and error-prone. This guide shows how to drive `chap eval` from a small bash script so a
whole batch of evaluations runs unattended and the results are easy to compare.

By the end you will be able to:

- loop `chap eval` over several models, configurations, or backtest settings,
- keep the outputs organised so each run is identifiable,
- collect all runs into a single comparison table with `chap export-metrics`.

## Prerequisites

- You can run a single evaluation with `chap eval` (see the
  [Monday Afternoon workshop material](../kigali-workshop/kigali-workshop-material/monday-afternoon.md)).
- For the configuration sweep below, it helps to have done the
  [Make your model configurable](configurable-model.md) tutorial.
- A working shell (bash or zsh) and the Chap CLI installed.

## The two building blocks

Every pattern in this guide is built from just two commands:

- **`chap eval`** runs one backtest for one (model, configuration, settings) combination
  and writes a single NetCDF (`.nc`) file.
- **`chap export-metrics`** reads many `.nc` files and writes one CSV with one row per
  evaluation and one column per metric.

The key idea that makes looping work: **encode what varies in the output filename**. If
every run writes to a distinct, descriptive `.nc` file, you never overwrite a result and
you can always tell the runs apart in the final table.

## Setup

Pick a dataset once and reuse it for every run. We use the Laos subset hosted on GitHub,
and write every output into a `results/` directory:

```bash
DATASET="https://raw.githubusercontent.com/dhis2-chap/chap-core/master/example_data/laos_subset.csv"
mkdir -p results
```

Throughout, we use small backtest settings (`--backtest-params.n-splits 2
--backtest-params.n-periods 1`) so the loops finish quickly. Increase them for real
comparisons.

## Pattern 1: compare several models

Put the model URLs in a bash array and loop over them. Each model writes its own output
file named after the model:

```bash
models=(
  "https://github.com/dhis2-chap/minimalist_example_uv"
  "https://github.com/chap-models/minimalist_configurable_model"
  "https://github.com/dhis2-chap/chap_auto_ewars"
)

for model in "${models[@]}"; do
  name=$(basename "$model")
  echo ">>> Evaluating $name"
  chap eval \
      --model-name "$model" \
      --dataset-csv "$DATASET" \
      --output-file "results/model_${name}.nc" \
      --backtest-params.n-splits 2 \
      --backtest-params.n-periods 1
done
```

This leaves you with one `.nc` per model in `results/`. See
[Collect and compare](#collect-and-compare) below to turn them into a table.

## Pattern 2: sweep a configuration parameter

This is where the [configurable model](configurable-model.md) pays off. To find a good
value for `n_lags`, loop over candidate values, write a configuration file for each, and
evaluate. The swept value goes into both the config file and the output filename:

```bash
model="https://github.com/chap-models/minimalist_configurable_model"

for n_lags in 0 1 3 6; do
  echo ">>> Evaluating n_lags=$n_lags"
  cat > "results/config_nlags_${n_lags}.yaml" <<EOF
user_option_values:
  n_lags: ${n_lags}
additional_continuous_covariates:
  - mean_temperature
EOF

  chap eval \
      --model-name "$model" \
      --dataset-csv "$DATASET" \
      --output-file "results/nlags_${n_lags}.nc" \
      --model-configuration-yaml "results/config_nlags_${n_lags}.yaml" \
      --backtest-params.n-splits 2 \
      --backtest-params.n-periods 1
done
```

You can nest loops to sweep two things at once -- for example `n_lags` against the set of
additional covariates -- just make sure every combination still maps to a unique output
filename.

## Pattern 3: sweep backtest settings

The same idea works for the backtest itself. Here we vary the forecast horizon
(`n-periods`) for a single model:

```bash
model="https://github.com/chap-models/minimalist_configurable_model"

for n_periods in 1 2 3; do
  echo ">>> Evaluating horizon=$n_periods"
  chap eval \
      --model-name "$model" \
      --dataset-csv "$DATASET" \
      --output-file "results/horizon_${n_periods}.nc" \
      --backtest-params.n-splits 2 \
      --backtest-params.n-periods "$n_periods"
done
```

## Collect and compare

Whatever you swept, gather every `.nc` file into a single comparison table. Build the
repeated `--input-files` arguments from a glob so you do not have to list each file by
hand:

```bash
args=()
for f in results/*.nc; do
  args+=(--input-files "$f")
done

chap export-metrics "${args[@]}" --output-file results/comparison.csv
```

The CSV has one row per evaluation, identified by its `filename` and `model_name`, with
columns for each metric (`mae`, `rmse`, `crps`, `coverage_10_90`, ...). Because you
encoded the swept value in each filename, you can read the effect of your sweep straight
off the table:

```bash
column -s, -t results/comparison.csv | less -S
```

Lower `mae`, `rmse`, and `crps` mean better accuracy; `coverage_10_90` should be close to
0.80. You can also plot any single run with `chap plot-backtest`:

```bash
chap plot-backtest results/nlags_3.nc results/nlags_3.html --plot-type evaluation_plot
```

## A reusable script

Combining the pieces, here is a small self-contained script that sweeps `n_lags` and
prints a comparison. It is safe to re-run: it skips any evaluation whose output already
exists, so an interrupted batch can be resumed.

```bash
#!/usr/bin/env bash
set -euo pipefail

DATASET="https://raw.githubusercontent.com/dhis2-chap/chap-core/master/example_data/laos_subset.csv"
MODEL="https://github.com/chap-models/minimalist_configurable_model"
mkdir -p results

for n_lags in 0 1 3 6; do
  out="results/nlags_${n_lags}.nc"
  if [[ -f "$out" ]]; then
    echo ">>> Skipping n_lags=$n_lags (already done)"
    continue
  fi
  echo ">>> Evaluating n_lags=$n_lags"
  cat > "results/config_nlags_${n_lags}.yaml" <<EOF
user_option_values:
  n_lags: ${n_lags}
additional_continuous_covariates:
  - mean_temperature
EOF
  chap eval \
      --model-name "$MODEL" \
      --dataset-csv "$DATASET" \
      --output-file "$out" \
      --model-configuration-yaml "results/config_nlags_${n_lags}.yaml" \
      --backtest-params.n-splits 2 \
      --backtest-params.n-periods 1
done

args=()
for f in results/nlags_*.nc; do args+=(--input-files "$f"); done
chap export-metrics "${args[@]}" --output-file results/comparison.csv

echo "Done. Comparison:"
column -s, -t results/comparison.csv
```

## Tips

- **`set -euo pipefail`** at the top of a script stops the whole batch on the first error,
  instead of silently continuing after a failed run.
- **Skip existing outputs** (the `[[ -f "$out" ]]` check) so you can resume an interrupted
  sweep without recomputing everything.
- **Keep configs next to results.** Writing `config_nlags_3.yaml` alongside `nlags_3.nc`
  makes each run self-documenting.
- **Run sequentially first.** Evaluations are CPU- and network-heavy; get the loop correct
  on a small sweep before scaling it up.
- **Record runs to MLflow** by adding `--run-config.track` to each `chap eval` if you want
  the batch tracked beyond the CSV.
