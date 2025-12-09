# Evaluation Workflow: Comparing Models with CLI

This guide walks through the complete workflow for evaluating models, visualizing results, and comparing metrics using the CHAP CLI.

## Overview

The workflow consists of three main steps:

1. **evaluate2**: Run a backtest and export results to NetCDF format
2. **plot-backtest**: Generate visualizations from evaluation results
3. **export-metrics**: Compare metrics across multiple evaluations in CSV format

## Prerequisites

- CHAP Core installed (see [CLI setup guide](chap-core-cli-setup.md))
- A dataset CSV file with disease case data
- A GeoJSON file with region polygons (optional, auto-discovered if named same as CSV)

## Step 1: Create an Evaluation

Use `evaluate2` to run a backtest on a model and export results to NetCDF format:

```bash
chap evaluate2 \
    --model-name https://github.com/dhis2-chap/minimalist_example_r \
    --dataset-csv ./data/vietnam_data.csv \
    --output-file ./results/model_a_eval.nc \
    --backtest-params.n-periods 3 \
    --backtest-params.n-splits 7
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model-name` | Model path or GitHub URL | Required |
| `--dataset-csv` | Path to CSV with disease data | Required |
| `--output-file` | Path for output NetCDF file | Required |
| `--backtest-params.n-periods` | Forecast horizon (periods ahead) | 3 |
| `--backtest-params.n-splits` | Number of train/test splits | 7 |
| `--backtest-params.stride` | Step size between splits | 1 |
| `--model-configuration-yaml` | Optional YAML with model config | None |

### GeoJSON Auto-Discovery

If your dataset is `vietnam_data.csv`, CHAP will automatically look for `vietnam_data.geojson` in the same directory.

## Step 2: Visualize the Evaluation

Use `plot-backtest` to generate visualizations from the evaluation results:

```bash
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

```bash
chap evaluate2 \
    --model-name https://github.com/dhis2-chap/chap_auto_ewars_weekly \
    --dataset-csv ./data/vietnam_data.csv \
    --output-file ./results/model_b_eval.nc \
    --backtest-params.n-periods 3 \
    --backtest-params.n-splits 7
```

## Step 4: Export and Compare Metrics

Use `export-metrics` to compute metrics from multiple evaluations and export to CSV:

```bash
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

```bash
chap export-metrics \
    --input-files ./results/model_a_eval.nc ./results/model_b_eval.nc \
    --output-file ./results/comparison.csv \
    --metric-ids rmse_aggregate mae_aggregate crps
```

## Complete Example

Here's a complete workflow comparing two models on Vietnam dengue data:

```bash
# Step 1: Evaluate first model
chap evaluate2 \
    --model-name https://github.com/dhis2-chap/minimalist_example_r \
    --dataset-csv ./vietnam_dengue.csv \
    --output-file ./eval_minimalist.nc

# Step 2: Plot first model results
chap plot-backtest \
    --input-file ./eval_minimalist.nc \
    --output-file ./plot_minimalist.html

# Step 3: Evaluate second model
chap evaluate2 \
    --model-name https://github.com/dhis2-chap/chap_auto_ewars_weekly \
    --dataset-csv ./vietnam_dengue.csv \
    --output-file ./eval_ewars.nc

# Step 4: Plot second model results
chap plot-backtest \
    --input-file ./eval_ewars.nc \
    --output-file ./plot_ewars.html

# Step 5: Compare metrics
chap export-metrics \
    --input-files ./eval_minimalist.nc ./eval_ewars.nc \
    --output-file ./model_comparison.csv

# View the comparison
cat ./model_comparison.csv
```

## Tips

- **Consistent parameters**: Use the same `n-periods` and `n-splits` when comparing models
- **Same dataset**: Always use identical datasets for fair comparison
- **Multiple runs**: Consider running evaluations with different random seeds for robustness
- **Metric interpretation**: Lower RMSE/MAE/CRPS is better; higher coverage ratios indicate better calibrated uncertainty
