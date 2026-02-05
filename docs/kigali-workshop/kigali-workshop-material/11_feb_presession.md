# Pre-session: Working with Datasets

This session covers how to prepare datasets for use with CHAP -- from downloading data through the Modeling App, to transforming and validating your own CSV files, to running an evaluation.

## Why datasets need to be CHAP-compatible

CHAP models are designed to be interchangeable: any model that follows the CHAP interface can be evaluated on any CHAP-compatible dataset. For this to work, datasets must follow a common CSV format so that models know which columns to expect and how to parse time periods and locations.

The required columns are:

| Column | Description |
|--------|-------------|
| `time_period` | Time period in `YYYY-MM` (monthly) or `YYYY-Wnn` (weekly) format |
| `location` | Location identifier matching the GeoJSON features |
| `disease_cases` | Observed case counts |

Additional columns (e.g. `rainfall`, `mean_temperature`, `population`) are used as covariates by models that need them. A matching GeoJSON file with region polygons is optional but recommended for spatial visualizations.

### Example CSV

```csv
time_period,rainfall,mean_temperature,disease_cases,population,location
2023-01,37.9,20.0,12,75000,Region_A
2023-02,8.5,22.2,8,75000,Region_A
2023-01,55.3,25.1,30,120000,Region_B
2023-02,12.1,26.8,22,120000,Region_B
```

## Creating and downloading a dataset from the Modeling App

If you have CHAP connected to a DHIS2 instance via the Modeling App, you can create and download datasets directly:

1. Open the DHIS2 Modeling App
2. Navigate to **Evaluation** and select your disease, indicator, and regions of interest
3. Configure the time period range
4. Run the evaluation -- the app will pull data from DHIS2 and climate sources
5. Download the resulting dataset as a CSV file from the app

The downloaded CSV will already be in CHAP-compatible format.

## Converting a Modeling App request to CSV and GeoJSON

If you have a JSON request payload from the DHIS2 Modeling App (the `create-backtest-with-data` format), you can convert it directly to a CHAP-compatible CSV and GeoJSON file pair using `chap convert-request`:

```bash
chap convert-request example_data/create-backtest-with-data.json /tmp/chap_convert_doctest
```

This reads the JSON file and produces two files:

- `/tmp/chap_convert_doctest.csv` -- a pivoted CSV with `time_period`, `location`, and feature columns
- `/tmp/chap_convert_doctest.geojson` -- the region boundaries extracted from the request

You can then validate the result:

```bash
chap validate --dataset-csv /tmp/chap_convert_doctest.csv
```

```bash
rm -f /tmp/chap_convert_doctest.csv /tmp/chap_convert_doctest.geojson
```

## Transforming data from other sources

If your data comes from a source other than DHIS2, you need to make sure it matches the CHAP format.

### Column names

Rename your columns to match the expected names:

- Time column must be named `time_period`
- Location column must be named `location`
- Case count column must be named `disease_cases`

### Time period format

Convert your dates to the correct format:

- **Monthly data**: `YYYY-MM` (e.g. `2023-01`, `2023-12`)
- **Weekly data**: `YYYY-Wnn` (e.g. `2023-W01`, `2023-W52`)

### Consecutive periods

All time periods must be consecutive with no gaps. Every location must have data for every time period in the dataset.

### GeoJSON file

If you want spatial visualizations, create a GeoJSON file where each feature's identifier matches the `location` values in your CSV. Name the GeoJSON file with the same base name as your CSV (e.g. `my_data.csv` and `my_data.geojson`) for automatic discovery.

### Example: transforming a pandas DataFrame

```python
import pandas as pd

# Suppose you have a DataFrame with different column names
df = pd.DataFrame({
    "date": ["2023-01-01", "2023-02-01", "2023-01-01", "2023-02-01"],
    "region": ["Region_A", "Region_A", "Region_B", "Region_B"],
    "cases": [12, 8, 30, 22],
    "rain_mm": [37.9, 8.5, 55.3, 12.1],
})

# Rename columns to match CHAP format
df = df.rename(columns={
    "region": "location",
    "cases": "disease_cases",
    "rain_mm": "rainfall",
})

# Convert dates to YYYY-MM format
df["time_period"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m")
df = df.drop(columns=["date"])

# Reorder columns
df = df[["time_period", "rainfall", "disease_cases", "location"]]
```

## Validating your dataset

Use the `chap validate` command to check that your CSV is CHAP-compatible before running an evaluation.

### Basic validation

```console
chap validate --dataset-csv my_data.csv
```

This checks for:

- Required columns (`time_period`, `location`)
- Missing values (NaN) in covariate columns
- Location completeness (every location has the same set of time periods)

### Validation against a model

You can also validate that your dataset has the covariates a specific model requires:

```console
chap validate \
    --dataset-csv my_data.csv \
    --model-name https://github.com/dhis2-chap/minimalist_example_r
```

This additionally checks that all required covariates for the model are present in the dataset, and that the time period type (weekly/monthly) matches what the model supports.

### Using a data source mapping

If your CSV uses different column names than what the model expects, provide a mapping file:

```console
chap validate \
    --dataset-csv my_data.csv \
    --model-name https://github.com/dhis2-chap/minimalist_example_r \
    --data-source-mapping mapping.json
```

Where `mapping.json` maps model covariate names to your CSV column names:

```json
{"rainfall": "rain_mm", "mean_temperature": "temp_avg"}
```

### Example: validating the bundled dataset

```bash
chap validate --dataset-csv example_data/laos_subset.csv
```

## Running an evaluation

Once your dataset is validated, you can evaluate a model on it using `chap eval`:

```bash
chap eval \
    --model-name external_models/naive_python_model_uv \
    --dataset-csv example_data/laos_subset.csv \
    --output-file ./eval_presession_doctest.nc \
    --backtest-params.n-splits 2 \
    --backtest-params.n-periods 1
```

Then visualize the results:

```bash
chap plot-backtest \
    --input-file ./eval_presession_doctest.nc \
    --output-file ./plot_presession_doctest.html
```

```bash
rm -f ./eval_presession_doctest.nc ./plot_presession_doctest.html
```

For more details on evaluation parameters, see the [Evaluation Workflow](../../chap-cli/evaluation-workflow.md) guide.
