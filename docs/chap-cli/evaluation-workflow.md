# Evaluation Workflow: Comparing Models with CLI

This guide walks through the complete workflow for evaluating models, visualizing results, and comparing metrics using the CHAP CLI.

## Overview

The workflow consists of three main steps:

1. **evaluate2**: Run a backtest and export results to NetCDF format
2. **plot-backtest**: Generate visualizations from evaluation results
3. **export-metrics**: Compare metrics across multiple evaluations in CSV format

## Prerequisites

- CHAP Core installed (see [Setup guide](chap-core-cli-setup.md))
- A dataset CSV file with disease case data
- A GeoJSON file with region polygons (optional, auto-discovered if named same as CSV)

## Verify Installation

Before starting, verify that the CLI tools are installed correctly:

```bash
chap evaluate2 --help
```

```bash
chap plot-backtest --help
```

```bash
chap export-metrics --help
```

## Example Dataset

CHAP includes a small example dataset for testing and learning:

- `example_data/laos_subset.csv` - Monthly dengue data for 3 provinces (2010-2012)
- `example_data/laos_subset.geojson` - Matching polygon boundaries

This dataset contains 108 rows with rainfall, temperature, disease cases, and population data for Bokeo, Vientiane, and Savannakhet provinces.

## Step 1: Create an Evaluation

Use `evaluate2` to run a backtest on a model and export results to NetCDF format.

### Standard Models (GitHub URL or Local Directory)

For models hosted on GitHub or cloned locally:

```console
chap evaluate2 \
    --model-name https://github.com/dhis2-chap/minimalist_example_r \
    --dataset-csv ./data/vietnam_data.csv \
    --output-file ./results/model_a_eval.nc \
    --backtest-params.n-periods 3 \
    --backtest-params.n-splits 7
```

Or using a local directory:

```console
chap evaluate2 \
    --model-name /path/to/minimalist_example_r \
    --dataset-csv ./data/vietnam_data.csv \
    --output-file ./results/model_a_eval.nc \
    --backtest-params.n-periods 3 \
    --backtest-params.n-splits 7
```

### Chapkit Models

Chapkit models are REST API-based models that follow the chapkit specification. See [Running models with chapkit](../external_models/chapkit.md) for more details.

**From a running chapkit service (URL):**

```console
chap evaluate2 \
    --model-name http://localhost:8000 \
    --dataset-csv ./data/vietnam_data.csv \
    --output-file ./results/chapkit_eval.nc \
    --run-config.is-chapkit-model \
    --backtest-params.n-periods 3 \
    --backtest-params.n-splits 7
```

**From a local chapkit model directory (auto-starts the service):**

When you provide a directory path with `--run-config.is-chapkit-model`, CHAP automatically:

1. Starts a FastAPI dev server from the model directory using `uv run fastapi dev`
2. Waits for the service to become healthy
3. Runs the evaluation
4. Stops the service when complete

```console
chap evaluate2 \
    --model-name /path/to/your/chapkit/model \
    --dataset-csv ./data/vietnam_data.csv \
    --output-file ./results/chapkit_eval.nc \
    --run-config.is-chapkit-model \
    --backtest-params.n-periods 3 \
    --backtest-params.n-splits 7
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model-name` | Model path, GitHub URL, or chapkit service URL | Required |
| `--dataset-csv` | Path to CSV with disease data | Required |
| `--output-file` | Path for output NetCDF file | Required |
| `--backtest-params.n-periods` | Forecast horizon (periods ahead) | 3 |
| `--backtest-params.n-splits` | Number of train/test splits | 7 |
| `--backtest-params.stride` | Step size between splits | 1 |
| `--model-configuration-yaml` | Optional YAML with model config | None |
| `--run-config.is-chapkit-model` | Flag to indicate chapkit model | false |
| `--run-config.ignore-environment` | Skip environment setup | false |
| `--run-config.debug` | Enable debug logging | false |
| `--run-config.run-directory-type` | Directory handling: `latest`, `timestamp`, or `use_existing` | timestamp |
| `--historical-context-years` | Years of historical data for plot context | 6 |

### GeoJSON Auto-Discovery

If your dataset is `vietnam_data.csv`, CHAP will automatically look for `vietnam_data.geojson` in the same directory.

## Step 2: Visualize the Evaluation

Use `plot-backtest` to generate visualizations from the evaluation results:

```console
chap plot-backtest \
    --input-file ./results/model_a_eval.nc \
    --output-file ./results/model_a_plot.html \
    --plot-type backtest_plot_1
```

### Available Plot Types

| Plot Type | Description |
|-----------|-------------|
| `backtest_plot_1` | Standard backtest visualization with forecasts vs observations |
| `evaluation_plot` | Evaluation summary plot |
| `ratio_of_samples_above_truth` | Shows forecast bias across locations |

### Output Formats

The output format is determined by file extension:

- `.html` - Interactive HTML (recommended)
- `.png` - Static PNG image
- `.svg` - Vector SVG image
- `.pdf` - PDF document
- `.json` - Vega JSON specification

## Step 3: Create Another Evaluation

Run the same process with a different model for comparison:

```console
chap evaluate2 \
    --model-name https://github.com/dhis2-chap/chap_auto_ewars_weekly \
    --dataset-csv ./data/vietnam_data.csv \
    --output-file ./results/model_b_eval.nc \
    --backtest-params.n-periods 3 \
    --backtest-params.n-splits 7
```

## Step 4: Export and Compare Metrics

Use `export-metrics` to compute metrics from multiple evaluations and export to CSV:

```console
chap export-metrics \
    --input-files ./results/model_a_eval.nc ./results/model_b_eval.nc \
    --output-file ./results/comparison.csv
```

### Output Format

The CSV contains one row per evaluation with metadata and metric columns:

```csv
filename,model_name,model_version,rmse_aggregate,mae_aggregate,crps,ratio_within_10th_90th,ratio_within_25th_75th,test_sample_count
model_a_eval.nc,minimalist_example_r,1.0.0,45.2,32.1,0.045,0.85,0.65,168
model_b_eval.nc,chap_auto_ewars_weekly,2.0.0,38.7,28.4,0.038,0.88,0.70,168
```

### Available Metrics

| Metric ID | Description |
|-----------|-------------|
| `rmse_aggregate` | Root Mean Squared Error (across all data) |
| `mae_aggregate` | Mean Absolute Error (across all data) |
| `crps` | Continuous Ranked Probability Score |
| `ratio_within_10th_90th` | Coverage ratio for 10th-90th percentile interval |
| `ratio_within_25th_75th` | Coverage ratio for 25th-75th percentile interval |
| `test_sample_count` | Number of test samples |

### Selecting Specific Metrics

To export only specific metrics:

```console
chap export-metrics \
    --input-files ./results/model_a_eval.nc ./results/model_b_eval.nc \
    --output-file ./results/comparison.csv \
    --metric-ids rmse_aggregate mae_aggregate crps
```

## Complete Example: Standard Models

Here's a complete workflow comparing two standard models using the included example dataset:

```console
# Step 1: Evaluate first model (auto-regressive)
chap evaluate2 \
    --model-name https://github.com/dhis2-chap/chap_auto_ewars \
    --dataset-csv ./example_data/laos_subset.csv \
    --output-file ./eval_ewars.nc \
    --backtest-params.n-splits 3

# Step 2: Plot first model results
chap plot-backtest \
    --input-file ./eval_ewars.nc \
    --output-file ./plot_ewars.html

# Step 3: Evaluate second model (minimalist R model)
chap evaluate2 \
    --model-name https://github.com/dhis2-chap/minimalist_example_r \
    --dataset-csv ./example_data/laos_subset.csv \
    --output-file ./eval_minimalist.nc \
    --backtest-params.n-splits 3

# Step 4: Plot second model results
chap plot-backtest \
    --input-file ./eval_minimalist.nc \
    --output-file ./plot_minimalist.html

# Step 5: Compare metrics
chap export-metrics \
    --input-files ./eval_ewars.nc ./eval_minimalist.nc \
    --output-file ./model_comparison.csv

# View the comparison
cat ./model_comparison.csv
```

The GeoJSON file `example_data/laos_subset.geojson` is automatically discovered since it has the same base name as the CSV.

## Complete Example: Chapkit Models

Here's a workflow using chapkit models, including both a running service and a local directory:

### Option A: Using a running chapkit service

First, start your chapkit model service (e.g., using Docker):

```console
docker run -p 8000:8000 ghcr.io/dhis2-chap/chtorch:latest
```

Then run the evaluation:

```console
# Evaluate the chapkit model
chap evaluate2 \
    --model-name http://localhost:8000 \
    --dataset-csv ./example_data/laos_subset.csv \
    --output-file ./eval_chapkit.nc \
    --run-config.is-chapkit-model \
    --backtest-params.n-splits 3

# Plot results
chap plot-backtest \
    --input-file ./eval_chapkit.nc \
    --output-file ./plot_chapkit.html
```

### Option B: Using a local chapkit model directory (auto-start)

If you have a chapkit model in a local directory, CHAP can automatically start and stop the service:

```console
# Clone or create your chapkit model
git clone https://github.com/your-org/your-chapkit-model /path/to/chapkit-model

# Evaluate with auto-start (CHAP starts the service automatically)
chap evaluate2 \
    --model-name /path/to/chapkit-model \
    --dataset-csv ./example_data/laos_subset.csv \
    --output-file ./eval_local_chapkit.nc \
    --run-config.is-chapkit-model \
    --backtest-params.n-splits 3

# Plot results
chap plot-backtest \
    --input-file ./eval_local_chapkit.nc \
    --output-file ./plot_local_chapkit.html
```

### Comparing chapkit and standard models

You can compare chapkit models with standard models using export-metrics:

```console
# Evaluate a standard model
chap evaluate2 \
    --model-name https://github.com/dhis2-chap/minimalist_example_r \
    --dataset-csv ./example_data/laos_subset.csv \
    --output-file ./eval_standard.nc \
    --backtest-params.n-splits 3

# Evaluate a chapkit model
chap evaluate2 \
    --model-name /path/to/chapkit-model \
    --dataset-csv ./example_data/laos_subset.csv \
    --output-file ./eval_chapkit.nc \
    --run-config.is-chapkit-model \
    --backtest-params.n-splits 3

# Compare both
chap export-metrics \
    --input-files ./eval_standard.nc ./eval_chapkit.nc \
    --output-file ./comparison.csv
```

## Tips

- **Consistent parameters**: Use the same `n-periods` and `n-splits` when comparing models
- **Same dataset**: Always use identical datasets for fair comparison
- **Multiple runs**: Consider running evaluations with different random seeds for robustness
- **Metric interpretation**: Lower RMSE/MAE/CRPS is better; higher coverage ratios indicate better calibrated uncertainty
- **Chapkit auto-start**: When using local chapkit directories, ensure `uv` is installed and the model directory has a valid FastAPI app structure with a `/health` endpoint
