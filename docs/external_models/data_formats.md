## Making a dataset CHAP-compliant
To use your data with CHAP, it must be provided as a CSV file that follows the format described below. A complete working example is available in [example_data/laos_subset.csv](https://github.com/dhis2-chap/chap-core/blob/master/example_data/laos_subset.csv).

### Required columns

Every dataset must contain these columns:

| Column | Description |
|--------|-------------|
| `time_period` | Time period identifier (see format below) |
| `location` | Location identifier (e.g. district name or code) |
| `disease_cases` | Number of observed disease cases (the target variable) |

### Covariate columns

Models declare which climate covariates they need (e.g. `rainfall`, `mean_temperature`). The CSV must include a column for each covariate required by the model you are running. Covariate columns must not contain missing or NaN values.

### Population column

Most models also require a `population` column with the population count for each location and time period. Check the model's documentation to confirm whether it is needed.

### Mapping custom column names

If your CSV uses different column names than those expected by a model, you can provide a JSON mapping file via the `--data-source-mapping` option instead of renaming columns. See the [eval command reference](../chap-cli/eval-reference.md#data-source-mapping) for details.

### Specific named fields

The table below summarises every column name that CHAP recognises:

| Column | Required | Description |
|--------|----------|-------------|
| `time_period` | Yes | Time period in ISO format |
| `location` | Yes | Location identifier |
| `disease_cases` | Yes | Observed case count (target) |
| `population` | Model-dependent | Population for the location |
| *covariate columns* | Model-dependent | Climate/environmental covariates (e.g. `rainfall`, `mean_temperature`) |

### Time period format

The `time_period` column uses:

- `YYYY-MM` format for monthly data (e.g., `2023-01`)
- `YYYY-Wnn` format for weekly data (e.g., `2023-W01`)

All time periods in a dataset must use the same frequency (do not mix monthly and weekly).

### Missing values

Different columns have different rules for missing (NaN) values:

| Column | NaN allowed? | Notes |
|--------|-------------|-------|
| `time_period` | No | Every row must have a valid period |
| `location` | No | Every row must have a location |
| `disease_cases` | Yes | Surveillance data may have gaps; CHAP tolerates NaN in the target |
| Covariates | No | Climate/environmental columns must be fully observed |
| `population` | Yes | Missing values are forward-filled by interpolation when present |

CHAP does not perform imputation on covariates. If your data pipeline produces NaN values in covariate columns, those must be resolved before passing the CSV to CHAP. Locations with missing covariate values will be rejected during dataset validation.

Every `(location, time_period)` combination in the dataset must be present as a row. Missing rows are not allowed -- if a location covers 12 months, it must have all 12 rows. CHAP will raise an error if any location has fewer time periods than others.

### Requirement for Periods to be consecutive

Time periods must be consecutive with no gaps. For example, monthly data that jumps from `2023-01` to `2023-03` (skipping February) is invalid.

In addition, every location must have exactly the same set of time periods. If your dataset covers three locations over 12 months, each location must have all 12 monthly rows.

### Multi-location example

Below is a complete example showing two locations with consecutive monthly periods:

```csv
time_period,location,disease_cases,rainfall,mean_temperature,population
2023-01,district_a,150,120.5,28.3,50000
2023-02,district_a,180,95.0,27.8,50000
2023-03,district_a,210,60.2,26.5,50000
2023-01,district_b,90,110.0,29.1,35000
2023-02,district_b,105,88.4,28.6,35000
2023-03,district_b,130,55.7,27.0,35000
```

### Monthly data example
```csv
time_period,rainfall,mean_temperature,disease_cases,location
2023-01,10,30,200,loc1
2023-02,2,30,100,loc1
```

### Weekly data example
```csv
time_period,rainfall,mean_temperature,disease_cases,location
2023-W01,12,28,45,loc1
2023-W02,8,29,52,loc1
```

### Validating your dataset

Use the `chap validate` command to check that a CSV file meets all the requirements described above before running an evaluation.

The command checks for:

- Missing or NaN values in covariate columns
- Consecutive time periods (no gaps)
- Location completeness (every location covers the same time periods)

The command exits with code 0 when no errors are found, or code 1 if any errors are detected.

#### Validating a correct dataset

```bash
chap validate --dataset-csv example_data/laos_subset.csv
```

#### Detecting missing covariate values

```bash
chap validate --dataset-csv example_data/faulty_datasets/missing_covariate_values.csv || true
```

#### Validating against a model

To also verify that the dataset has the covariates and period type required by a specific model, pass `--model-name`:

```bash
chap validate \
    --dataset-csv example_data/laos_subset.csv \
    --model-name external_models/naive_python_model_uv
```

#### Mapping custom column names

If your CSV uses non-standard column names, supply a mapping file with `--data-source-mapping`:

```console
chap validate --dataset-csv ./my_data.csv \
    --model-name https://github.com/dhis2-chap/minimalist_example_r \
    --data-source-mapping ./column_mapping.json
```
