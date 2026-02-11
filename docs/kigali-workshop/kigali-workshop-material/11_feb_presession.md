# Pre-session: Working with Datasets

This session covers how to prepare datasets for use with CHAP -- from downloading data through the Modeling App, to transforming and validating your own CSV files, to running an evaluation.

## Why datasets need to be CHAP-compatible

CHAP models are designed to be interchangeable: any model that follows the CHAP interface can be evaluated on any CHAP-compatible dataset. For this to work, datasets must follow a common CSV format so that models know which columns to expect and how to parse time periods and locations.

The required columns are:

| Column          | Description                                                      |
| --------------- | ---------------------------------------------------------------- |
| `time_period`   | Time period in `YYYY-MM` (monthly) or `YYYY-Wnn` (weekly) format |
| `location`      | Location identifier matching the GeoJSON features                |
| `disease_cases` | Observed case counts                                             |

Additional columns (e.g. `rainfall`, `mean_temperature`, `population`) are used as covariates by models that need them. A matching GeoJSON file with region polygons is optional but recommended for spatial visualizations.

### Example CSV

```csv
time_period,rainfall,mean_temperature,disease_cases,population,location
2023-01,37.9,20.0,12,75000,Region_A
2023-02,8.5,22.2,8,75000,Region_A
2023-01,55.3,25.1,30,120000,Region_B
2023-02,12.1,26.8,22,120000,Region_B
```

## Optional: Extracting Climate & Environmental Data for Modelling in DHIS2

This short guide describes how to import climate and environmental data, configure a model in the DHIS2 Modelling App, and extract the modelling payload.

### 1. Import data using the Climate App

Use the Climate App to import climate and environmental indicators at the **same organisational level and period type** (weekly or monthly) as your disease data. Indicators of interest include air temperature, CHIRPS precipitation (or ERA5-Land precipitation if you know this performs better), relative humidity, NDVI (vegetation), and urban/built-up areas. You may also include other disease-relevant indicators such as soil moisture, surface water, land surface temperature, or elevation. Ensure all imported data are available as data elements.

### 2. Run analytics

After importing the data, run analytics in DHIS2.

### 3. Open the Modelling App

Open the Modelling App and confirm you are using version **4.0.0** or later.

### 4. Create a model

Go to Models, click New model, and select **CHAP-EWARS Model**. This model supports additional covariates. Give the model a clear name such as `extract_data`. Leave n_lags, precision, and regional seasonal settings unchanged.

### 5. Add covariates

Add all covariates you imported via the Climate App by typing their names and using underscores instead of spaces (for example `NDVI` or `relative_humidity`). Save the model when finished.

### 6. Open "Create an evaluation form"

Go to "Overview" and click "New evaluation. Select the period type, date range, organisation units, and the model you just created. Open "Dataset Configuration" and map each covariate to its corresponding data element you just imported data to. Save the configuration.

### 7. Run a dry run

Click "Start dry run" to verify that the data and configuration are accepted. Continue only if the dry run succeeds.

### 8. Download the payload

Click **Download request** to save the modelling payload to your computer as JSON-file.

<img src="../assets/download-button.png" alt="Download request button">

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

An example of how to do this with climate tools is here https://climate-tools.dhis2.org/guides/import-chap/harmonize-to-chap/

## Validating your dataset

Use the `chap validate` command to check that your CSV is CHAP-compatible before running an evaluation.

### Basic validation

```bash
chap validate --dataset-csv example_data/laos_subset.csv
```

This checks for:

- Required columns (`time_period`, `location`)
- Missing values (NaN) in covariate columns
- Location completeness (every location has the same set of time periods)

### Validation against a model

You can also validate that your dataset has the covariates a specific model requires:

```bash
chap validate \
    --dataset-csv example_data/laos_subset.csv \
    --model-name external_models/naive_python_model_uv
```

This additionally checks that all required covariates for the model are present in the dataset, and that the time period type (weekly/monthly) matches what the model supports.

### Using a data source mapping

If your CSV uses different column names than what the model expects, provide a mapping file:

```bash
chap validate \
    --dataset-csv example_data/laos_subset_custom_columns.csv \
    --data-source-mapping example_data/column_mapping.json
```

Where `column_mapping.json` maps model covariate names to your CSV column names:

```json
{ "rainfall": "rain_mm", "mean_temperature": "temp_avg" }
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
