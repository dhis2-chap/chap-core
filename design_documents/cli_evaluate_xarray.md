# CLI evaluate2 Command with xarray/NetCDF Output

## Overview

Create a new `evaluate2` CLI command that evaluates a single model and exports results to NetCDF format using xarray. This leverages the recently added Evaluation abstraction to provide a clean, standardized output format for model evaluation results.

## User Requirements

- New separate command (keep existing `evaluate` command unchanged)
- Single model evaluation only (no comma-separated models)
- Output format: NetCDF (.nc) using xarray
- Input: CSV files (same as current evaluate command)
- No CSV output (xarray/NetCDF only)
- Adapt `from_samples_with_truth` signature instead of creating new method

## Implementation Approach

### 1. Add Serialization Methods to Evaluation Class

**File:** `chap_core/assessment/evaluation.py`

Add two new methods and two helper functions:

#### Methods to add:

**`to_file(filepath, model_name=None, model_configuration=None, model_version=None)`**
- Export evaluation to NetCDF using xarray
- Convert FlatEvaluationData to xarray.Dataset structure
- Store model metadata as global attributes (JSON-encoded)
- Write to file using `ds.to_netcdf()`

**`from_file(filepath) -> Evaluation` (classmethod)**
- Load evaluation from NetCDF file
- Read xarray.Dataset and convert back to FlatEvaluationData
- Create in-memory BackTest object (without database persistence)
- Return Evaluation instance

**No changes needed to `from_samples_with_truth()` signature:**
- Keep existing signature unchanged
- In CLI, create ConfiguredModelDB and ModelTemplateDB objects from ModelTemplate
- Pass these objects to `from_samples_with_truth()` just like REST API does
- This maintains consistency between CLI and REST API workflows

#### Helper functions:

**`_flat_data_to_xarray(flat_data, model_metadata) -> xr.Dataset`**
- Convert FlatEvaluationData to xarray.Dataset
- Structure: 3D forecast array (location, time_period, sample)
- Structure: 2D observed array (location, time_period)
- Structure: 2D horizon_distance array (location, time_period)
- Add model metadata as global attributes

**`_xarray_to_flat_data(ds) -> FlatEvaluationData`**
- Convert xarray.Dataset back to FlatEvaluationData
- Reconstruct FlatForecasts and FlatObserved DataFrames
- Handle NaN values (missing data)

### 2. xarray.Dataset Structure

```
<xarray.Dataset>
Dimensions:
    location: N_locations
    time_period: N_periods
    sample: N_samples

Coordinates:
    location (location): string array of org_unit IDs
    time_period (time_period): string array of period IDs
    sample (sample): integer array [0, 1, ..., N_samples-1]

Data Variables:
    forecast (location, time_period, sample): float64
    horizon_distance (location, time_period): int64
    observed (location, time_period): float64

Global Attributes:
    title: "CHAP Model Evaluation Results"
    model_name: string
    model_configuration: JSON string
    model_version: string
    created_date: ISO timestamp
    split_periods: JSON array
    org_units: JSON array
    chap_version: string
```

**Rationale:** Multi-dimensional structure is natural for xarray and efficient for NetCDF. Coordinates enable labeled indexing. Global attributes provide complete traceability.

### 3. Add evaluate2 CLI Command

**File:** `chap_core/cli.py`

Add new command with signature:

```python
@app.command()
def evaluate2(
    model_name: str,
    output_file: Path,
    dataset_csv: Optional[Path] = None,
    dataset_name: Optional[DataSetType] = None,
    dataset_country: Optional[str] = None,
    polygons_json: Optional[Path] = None,
    polygons_id_field: str = "id",
    prediction_length: int = 6,
    n_splits: int = 7,
    stride: int = 1,
    ignore_environment: bool = False,
    debug: bool = False,
    log_file: Optional[str] = None,
    run_directory_type: Optional[Literal["latest", "timestamp", "use_existing"]] = "timestamp",
    model_configuration_yaml: Optional[str] = None,
    is_chapkit_model: bool = False,
)
```

#### Implementation steps:

1. Initialize logging (reuse existing helper)
2. Validate single model only (reject if comma-separated)
3. Load dataset using `_load_dataset()` helper (already exists)
4. Load ModelTemplate using `ModelTemplate.from_directory_or_github_url()`
5. Create in-memory ModelTemplateDB and ConfiguredModelDB objects from ModelTemplate
6. Get configured model instance using `template.get_model()`
7. Run backtest using `backtest()` from prediction_evaluator.py
8. Create BackTestCreate info object
9. Create Evaluation using `Evaluation.from_samples_with_truth()` (with ConfiguredModelDB)
10. Export using `evaluation.to_file()` with model metadata
11. Log completion

**Pattern to follow:** Similar to `db_worker_functions.run_backtest()` but with in-memory database objects instead of persisted ones.

### 4. Data Flow

```
CSV → DataSet
         ↓
   ModelTemplate.from_directory_or_github_url()
         ↓
   Create ModelTemplateDB (in-memory)
         ↓
   Create ConfiguredModelDB (in-memory)
         ↓
   template.get_model() → configured model
         ↓
   backtest() → Iterable[DataSet[SamplesWithTruth]]
         ↓
   Create BackTestCreate info object
         ↓
   Evaluation.from_samples_with_truth(configured_model, info)
         ↓
   Evaluation object
         ↓
   evaluation.to_flat() → FlatEvaluationData
         ↓
   _flat_data_to_xarray() → xarray.Dataset
         ↓
   ds.to_netcdf(output_file)
```

### 5. Testing Strategy

**File:** `tests/evaluation/test_evaluation_serialization.py` (new file)

Tests:
- `test_to_file_creates_netcdf()` - Verify file creation and structure
- `test_to_file_includes_metadata()` - Verify metadata in global attributes
- `test_from_file_loads_evaluation()` - Verify loading and data access
- `test_roundtrip_preserves_data()` - Verify save/load preserves all data
- `test_to_file_handles_edge_cases()` - NaN values, empty datasets, etc.

Use existing fixtures: `backtest`, `backtest_weeks`, `backtest_weeks_large` from conftest.py

**File:** `tests/test_cli_evaluate2.py` (new file)

Tests:
- `test_evaluate2_basic()` - Basic command execution
- `test_evaluate2_rejects_multiple_models()` - Validation error
- `test_evaluate2_with_config()` - Config file handling
- `test_evaluate2_output_structure()` - Verify NetCDF structure

Use existing fixtures: `nicaragua_path`, `tmp_path`

### 6. Branch and PR Strategy

**Branch name:** `feat/cli-evaluate2-xarray-export`

**Commit structure:**
1. `feat: add xarray serialization to Evaluation class`
2. `feat: add evaluate2 CLI command with NetCDF output`
3. `test: add tests for Evaluation serialization`
4. `test: add tests for evaluate2 CLI command`

**PR title:** `feat: add evaluate2 CLI command with xarray/NetCDF export`

**PR description template:**
```
## Summary
- Add to_file/from_file methods to Evaluation class for NetCDF serialization
- Adapt from_samples_with_truth to work without database objects
- Add new evaluate2 CLI command that outputs xarray/NetCDF format

## Motivation
Provide standardized NetCDF export format for evaluation results, enabling better integration with scientific analysis tools and easier result sharing.

## Changes
- chap_core/assessment/evaluation.py: Add serialization methods, adapt from_samples_with_truth
- chap_core/cli.py: Add evaluate2 command
- tests/: Add comprehensive test coverage

## Testing
- All new functionality covered by tests
- Existing tests pass (no regressions)
- Manual testing with sample datasets

## Usage
```bash
chap evaluate2 naive_model results.nc --dataset-csv data.csv --n-splits 10
```
```

## Critical Files

1. **`chap_core/assessment/evaluation.py`** - Add to_file(), from_file(), adapt from_samples_with_truth()
2. **`chap_core/cli.py`** - Add evaluate2 command
3. **`tests/evaluation/test_evaluation_serialization.py`** - New test file for serialization
4. **`tests/test_cli_evaluate2.py`** - New test file for CLI command

## Success Criteria

- evaluate2 command executes successfully
- NetCDF file has correct xarray.Dataset structure
- Model metadata preserved in global attributes
- Round-trip (save/load) preserves data integrity
- All tests pass: `make test`
- Lint passes: `make lint`
- No emojis in commits/code (per CLAUDE.md)
- Conventional commit format followed

## Usage Examples

### Basic evaluation
```bash
chap evaluate2 naive_model results.nc --dataset-csv my_data.csv
```

### With custom configuration
```bash
chap evaluate2 my_model output.nc \
  --dataset-csv data.csv \
  --model-configuration-yaml config.yaml \
  --n-splits 10 \
  --prediction-length 8
```

### Loading results in Python
```python
from chap_core.assessment.evaluation import Evaluation
import xarray as xr

# Option 1: Use xarray directly
ds = xr.open_dataset('results.nc')
print(ds.attrs['model_name'])
print(ds['forecast'].shape)
print(ds['observed'].values)

# Option 2: Use Evaluation abstraction
evaluation = Evaluation.from_file('results.nc')
flat_data = evaluation.to_flat()
print(flat_data.forecasts._df.head())
print(flat_data.observations._df.head())
```

## Comparison with Existing evaluate Command

| Feature | evaluate (existing) | evaluate2 (new) |
|---------|-------------------|-----------------|
| Output format | CSV (summary + detailed) | NetCDF (.nc) |
| Multiple models | Yes (comma-separated) | No (single model only) |
| Data structure | Flat tables | Multi-dimensional xarray |
| Metadata | Separate files | Embedded in NetCDF |
| Scientific tools | Limited support | Full support (xarray ecosystem) |
| File size | Larger (text-based) | Smaller (binary, compressed) |
| Human readable | Yes (CSV) | No (binary), but viewable with tools |

## Future Extensions

- Add support for quantile-based forecasts (not just samples)
- Support for additional output formats (Zarr, HDF5)
- Compression options for large datasets
- Metadata validation and schema versioning
- Integration with visualization tools