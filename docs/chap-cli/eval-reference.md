# eval Command Reference

The `eval` command runs a rolling-origin backtest evaluation on a disease prediction model and exports results in NetCDF format for analysis with scientific tools.

## Synopsis

```console
chap eval --model-name <MODEL> --dataset-csv <CSV_FILE> --output-file <OUTPUT.nc> [OPTIONS]
```

## Description

This command evaluates a single model by:

1. Loading disease and climate data from a CSV file
2. Auto-discovering GeoJSON polygon boundaries (same name as CSV with `.geojson` extension)
3. Splitting historical data into multiple train/test sets using rolling-origin backtesting
4. Training the model on each training set and generating probabilistic forecasts
5. Comparing forecasts against actual observations
6. Exporting results (predictions, observations, metrics) to NetCDF format

The output NetCDF file can be used with `plot-backtest` for visualization and `export-metrics` for metric extraction.

## Required Parameters

| Parameter | Description |
|-----------|-------------|
| `--model-name` | Model identifier. Can be a local directory path, GitHub URL, or chapkit service URL |
| `--dataset-csv` | Path to CSV file containing disease data |
| `--output-file` | Path for output NetCDF file (use `.nc` extension) |

## Backtest Parameters

Control how the evaluation splits and processes data:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--backtest-params.n-periods` | Forecast horizon (number of periods to predict ahead) | 3 |
| `--backtest-params.n-splits` | Number of train/test splits for cross-validation | 7 |
| `--backtest-params.stride` | Step size (in periods) between consecutive splits | 1 |

### Understanding Backtest Parameters

- **n-periods**: How far ahead the model forecasts. For monthly data, `n-periods=3` means 3-month forecasts.
- **n-splits**: How many times to train and evaluate. More splits = more robust evaluation but longer runtime.
- **stride**: How much to advance between splits. `stride=1` means every period gets a forecast; `stride=2` skips every other period.

Example: With 36 months of data, `n-periods=3`, `n-splits=7`, `stride=1`:
- Split 1: Train on months 1-26, test on months 27-29
- Split 2: Train on months 1-27, test on months 28-30
- ... and so on for 7 splits

## Run Configuration

Control model execution environment:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--run-config.is-chapkit-model` | Set when using a chapkit REST API model | false |
| `--run-config.ignore-environment` | Skip automatic environment setup | false |
| `--run-config.debug` | Enable verbose debug logging | false |
| `--run-config.log-file` | Path to write log output | None |
| `--run-config.run-directory-type` | Directory handling: `latest`, `timestamp`, or `use_existing` | timestamp |

### Run Directory Types

- **timestamp**: Create a new timestamped directory for each run (recommended for tracking)
- **latest**: Reuse the most recent run directory
- **use_existing**: Use existing run directory without modification

## Additional Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model-configuration-yaml` | Path to YAML file with model-specific parameters | None |
| `--historical-context-years` | Years of historical data to include for plotting context | 6 |
| `--data-source-mapping` | Path to JSON file for column name mapping | None |

## Data Source Mapping

When your CSV column names don't match the names expected by the model, use `--data-source-mapping` to provide a JSON mapping file.

### Format

The JSON file maps model covariate names (keys) to CSV column names (values):

```json
{
  "model_covariate_name": "csv_column_name"
}
```

### Example

If your model expects a column named `rainfall` but your CSV has `precipitation_mm`:

**mapping.json:**
```json
{
  "rainfall": "precipitation_mm",
  "mean_temperature": "temp_avg_celsius"
}
```

**Usage:**
```console
chap eval \
    --model-name ./my_model \
    --dataset-csv ./data.csv \
    --output-file ./eval.nc \
    --data-source-mapping ./mapping.json
```

### Common Use Cases

- Adapting DHIS2 exports with different column naming conventions
- Using the same model with datasets from different sources
- Renaming columns without modifying the original CSV

## Model Types

### Standard Models (GitHub or Local)

Models that follow the CHAP model specification can be loaded from:

- **GitHub URL**: `https://github.com/dhis2-chap/minimalist_example_r`
- **Local directory**: `/path/to/model` or `./model`

```console
chap eval \
    --model-name https://github.com/dhis2-chap/minimalist_example_r \
    --dataset-csv ./data/vietnam.csv \
    --output-file ./eval.nc
```

### Chapkit Models (REST API)

Chapkit models are REST API-based prediction services. Use `--run-config.is-chapkit-model` flag:

**From a running service:**
```console
chap eval \
    --model-name http://localhost:8000 \
    --dataset-csv ./data/vietnam.csv \
    --output-file ./eval.nc \
    --run-config.is-chapkit-model
```

**From a local directory (auto-starts the service):**
```console
chap eval \
    --model-name /path/to/chapkit-model \
    --dataset-csv ./data/vietnam.csv \
    --output-file ./eval.nc \
    --run-config.is-chapkit-model
```

When providing a directory path with `--run-config.is-chapkit-model`, CHAP automatically:
1. Starts a FastAPI dev server using `uv run fastapi dev`
2. Waits for the service to become healthy
3. Runs the evaluation
4. Stops the service when complete

## Input Data Format

### CSV File Requirements

The CSV file must contain:

| Column | Description |
|--------|-------------|
| `time_period` | Time period identifier (ISO format: YYYY-MM for monthly, YYYY-Www for weekly) |
| `location` | Location identifier matching GeoJSON feature IDs |
| `disease_cases` | Number of disease cases (target variable) |
| Climate covariates | Model-specific columns (e.g., `rainfall`, `mean_temperature`) |
| `population` | (Optional) Population for the location |

**Example CSV:**
```csv
time_period,location,disease_cases,rainfall,mean_temperature,population
2020-01,region_1,45,120.5,28.3,50000
2020-01,region_2,32,98.2,27.1,35000
2020-02,region_1,52,145.8,29.1,50000
...
```

### GeoJSON Auto-Discovery

CHAP automatically looks for a GeoJSON file with the same base name as the CSV:
- CSV: `vietnam_data.csv`
- GeoJSON: `vietnam_data.geojson` (auto-discovered in the same directory)

The GeoJSON features must have `id` properties matching the `location` values in the CSV.

## Output Format

The output is a NetCDF file containing:

- **Predictions**: Probabilistic forecasts at multiple quantiles (0.025, 0.25, 0.5, 0.75, 0.975)
- **Observations**: Actual disease case counts
- **Metadata**: Model name, version, configuration, evaluation parameters
- **Dimensions**: time, location, quantile, split

Use `plot-backtest` to visualize and `export-metrics` to extract metrics from the output file.

## Examples

### Basic Evaluation

```console
chap eval \
    --model-name https://github.com/dhis2-chap/chap_auto_ewars \
    --dataset-csv ./example_data/laos_subset.csv \
    --output-file ./eval_ewars.nc
```

### Custom Backtest Parameters

```console
chap eval \
    --model-name ./my_model \
    --dataset-csv ./data.csv \
    --output-file ./eval.nc \
    --backtest-params.n-periods 6 \
    --backtest-params.n-splits 12 \
    --backtest-params.stride 2
```

### With Model Configuration

```console
chap eval \
    --model-name https://github.com/dhis2-chap/minimalist_example_r \
    --dataset-csv ./data.csv \
    --output-file ./eval.nc \
    --model-configuration-yaml ./model_config.yaml
```

### Debug Mode with Logging

```console
chap eval \
    --model-name ./my_model \
    --dataset-csv ./data.csv \
    --output-file ./eval.nc \
    --run-config.debug \
    --run-config.log-file ./evaluation.log
```

### Complete Workflow

```console
# Step 1: Evaluate the model
chap eval \
    --model-name https://github.com/dhis2-chap/chap_auto_ewars \
    --dataset-csv ./data/vietnam.csv \
    --output-file ./results/ewars_eval.nc \
    --backtest-params.n-splits 10

# Step 2: Generate visualization
chap plot-backtest \
    --input-file ./results/ewars_eval.nc \
    --output-file ./results/ewars_plot.html

# Step 3: Export metrics
chap export-metrics \
    --input-files ./results/ewars_eval.nc \
    --output-file ./results/metrics.csv
```

## Related Commands

- [plot-backtest](evaluation-workflow.md#step-2-visualize-the-evaluation) - Generate visualizations from evaluation results
- [export-metrics](evaluation-workflow.md#step-4-export-and-compare-metrics) - Export and compare metrics across evaluations

## See Also

- [Evaluation Workflow](evaluation-workflow.md) - Complete guide to the evaluation workflow
- [Setup Guide](chap-core-cli-setup.md) - Installation and configuration
- [Chapkit Models](../external_models/chapkit.md) - Running models with chapkit
