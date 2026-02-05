#!/usr/bin/env python3
"""Generate extended test data with multiple locations and 3 years for integration testing."""

import json
import random
from pathlib import Path

# Load existing test data
input_file = Path(__file__).parent.parent / "example_data" / "create-backtest-with-data.json"
output_file = Path(__file__).parent.parent / "example_data" / "create-backtest-with-data-extended.json"

with open(input_file) as f:
    data = json.load(f)

# Extract existing location's geometry for reference
existing_feature = data["geojson"]["features"][0]
existing_location_id = existing_feature["id"]

# Create additional locations (need at least 5 for sklearn GroupKFold with n_splits=5)
additional_locations = [
    ("kJq2mPyFEHo", -2.5, 6.0),
    ("mNp3qRsFGHi", -3.0, 6.5),
    ("oQr4sTuVWXj", -3.5, 7.0),
    ("pRs5tUvWXYk", -4.0, 7.5),
]

for loc_id, lon_offset, lat_offset in additional_locations:
    feature = {
        "id": loc_id,
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [lon_offset, lat_offset],
                [lon_offset, lat_offset + 0.5],
                [lon_offset - 0.5, lat_offset + 0.5],
                [lon_offset - 0.5, lat_offset],
                [lon_offset, lat_offset]
            ]]
        },
        "properties": {
            "id": loc_id,
            "parent": "GhGOR2MfRFC",
            "parentGraph": "GhGOR2MfRFC",
            "level": 2
        }
    }
    data["geojson"]["features"].append(feature)

# Collect existing data by period and feature for the first location
existing_data = {}
for entry in data["providedData"]:
    key = (entry["featureName"], entry["period"])
    existing_data[key] = entry["value"]

# Generate 2021 data for first location (extrapolate from 2022 with slight variation)
new_data_entries = []
random.seed(42)  # For reproducibility

for month in range(1, 13):
    period_2021 = f"2021{month:02d}"
    period_2022 = f"2022{month:02d}"

    # Get 2022 values and create 2021 values with slight variation
    for feature in ["rainfall", "mean_temperature", "disease_cases"]:
        key_2022 = (feature, period_2022)
        if key_2022 in existing_data:
            base_value = existing_data[key_2022]
            if feature == "disease_cases":
                # Integer with +/- 10% variation
                new_value = int(base_value * random.uniform(0.9, 1.1))
            else:
                # Float with +/- 10% variation
                new_value = round(base_value * random.uniform(0.9, 1.1), 2)

            new_data_entries.append({
                "featureName": feature,
                "orgUnit": existing_location_id,
                "period": period_2021,
                "value": new_value
            })

# Generate all data for additional locations (2021-2023)
for loc_id, _, _ in additional_locations:
    for year in [2021, 2022, 2023]:
        for month in range(1, 13):
            period = f"{year}{month:02d}"
            # Use 2022 or 2023 data as reference
            ref_year = 2022 if year == 2021 else year
            ref_period = f"{ref_year}{month:02d}"

            for feature in ["rainfall", "mean_temperature", "disease_cases"]:
                key = (feature, ref_period)
                if key in existing_data:
                    base_value = existing_data[key]
                    # Add variation for each location
                    if feature == "disease_cases":
                        new_value = int(base_value * random.uniform(0.8, 1.2))
                    else:
                        new_value = round(base_value * random.uniform(0.85, 1.15), 2)

                    new_data_entries.append({
                        "featureName": feature,
                        "orgUnit": loc_id,
                        "period": period,
                        "value": new_value
                    })

# Add new entries to providedData
data["providedData"].extend(new_data_entries)

# Update name to reflect extended data
all_location_ids = [existing_location_id] + [loc_id for loc_id, _, _ in additional_locations]
data["name"] = f"Test Extended ({len(all_location_ids)} locations, 3 years)"

# Write output
with open(output_file, "w") as f:
    json.dump(data, f, indent=2)

print(f"Generated extended test data: {output_file}")
print(f"Locations: {len(all_location_ids)} ({', '.join(all_location_ids)})")
print(f"Years: 2021-2023 (36 months)")
print(f"Total data entries: {len(data['providedData'])}")
