# Run models in Chap with your own data

### Column names

Rename your columns to match the expected names:

- Time column must be named `time_period`
- Location column must be named `location`
- Case count column must be named `disease_cases`

For a complete example of a valid Chap CSV, see the [Laos dengue dataset](https://raw.githubusercontent.com/dhis2/climate-health-data/main/lao/chap_LAO_admin1_monthly.csv). In addition to the required columns, this file also includes the optional covariate columns: **population**, **location_name**, **rainfall**, **mean_temperature**, and **mean_relative_humidity**.

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

# Rename columns to match Chap format
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

An example of how to do this with climate tools is [here](https://climate-tools.dhis2.org/guides/import-chap/harmonize-to-chap/).

---

<div style="margin-bottom: 10rem;" markdown>

Next: [Validate your data](validate-data.md)

</div>