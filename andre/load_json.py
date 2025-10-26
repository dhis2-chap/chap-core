import argparse
import json
import re
import pandas as pd
from chap_core.rest_api_src.v1.routers.crud import BackTestFull

# ----------------------------------------
# Helpers: CamelCase → snake_case conversion

_camel1 = re.compile(r"(.)([A-Z][a-z]+)")  
_camel2 = re.compile(r"([a-z0-9])([A-Z])")  

def camel_to_snake(name: str) -> str: 
    """
    Convert a CamelCase string to snake_case.
    """
    s1 = _camel1.sub(r"\1_\2", name)
    return _camel2.sub(r"\1_\2", s1).lower()

def convert_keys(obj):
    """
    Recursively convert all dict keys in obj from CamelCase to snake_case.
    For lists, applies conversion to each element.
    """
    if isinstance(obj, dict):
        return {camel_to_snake(k): convert_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys(v) for v in obj]
    else:
        return obj

# ----------------------------------------
# Load and transform forecast JSON ➔ DataFrame

def load_forecasts(path: str) -> pd.DataFrame:
    """
    Read BackTestFull JSON from `path`, validate via Pydantic,
    and return a long-form DataFrame with one row per forecast step.
    """
    # 1. Load raw JSON
    with open(path) as f:
        raw = json.load(f)

    # 2. Convert all keys to snake_case
    snake = convert_keys(raw)
    # 3. Rename `model_id` key to `estimator_id` for Pydantic model
    snake["estimator_id"] = snake.pop("model_id")

    # 4. Validate / parse into Pydantic model
    backtest = BackTestFull.sqlmodel_validate(snake)

    # 5. Expand nested forecasts into flat records
    records = []
    for f in backtest.forecasts:
        # Use `period_id` (YYYYMM string) for each forecast origin
        period = f.period_id

        # Enumerate predicted values as 1‑month, 2‑month, ... ahead
        for step, val in enumerate(f.values, start=1):
            records.append({
                "period":    period,
                "step":      step,
                "predicted": val,
            })

    # 6. Build DataFrame and parse dates
    df_pred = pd.DataFrame(records)
    df_pred["period_dt"] = pd.to_datetime(df_pred["period"], format="%Y%m")
    return df_pred

# ----------------------------------------
# Load and transform gold (actual) JSON ➔ DataFrame
# ----------------------------------------
def load_gold(path: str) -> pd.DataFrame:
    """
    Read actual-case JSON from `path`, convert keys, and return DataFrame.
    """
    # 1. Load raw JSON
    with open(path) as f:
        raw = json.load(f)
        #print(raw)
    # 2. Convert keys to snake_case
    snake = convert_keys(raw)

    # 3. Build DataFrame from `data` list
    df = pd.DataFrame(snake["data"])

    # 4. Rename columns for clarity
    df = df.rename(columns={
        "pe":    "period",
        "ou":    "org_unit",
        "value": "actual"
    })

    # 5. Parse period string (YYYYMM) into datetime
    df["period_dt"] = pd.to_datetime(df["period"], format="%Y%m")
    return df

# ----------------------------------------
# Script entry point: parse args, load, compare
# ----------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load dengue forecasts & actuals, compare them"
    )
    parser.add_argument(
        "-p", "--pred", default="response.json",
        help="Path to BackTestFull JSON for forecasts"
    )
    parser.add_argument(
        "-g", "--gold", default="response_actual.json",
        help="Path to JSON file with actual dengue-case values"
    )
    parser.add_argument(
        "-o", "--output", default="comparison.csv",
        help="Output CSV for merged comparison"
    )
    args = parser.parse_args()

    # Load dataframes
    df_pred = load_forecasts(args.pred)
    df_gold = load_gold(args.gold)

    # Display first few rows for inspection
    print("Forecasts:")
    print(df_pred.head(), "\n")
    print("Gold actuals:")
    print(df_gold.head(), "\n")

    # Merge on `period` (and optionally `step` if needed)
    df_compare = pd.merge(
        df_pred, df_gold,
        on=["period"],
        how="inner",
        suffixes=("_pred","_gold")
    )

    # Show result and optionally save to CSV
    print("Comparison:")
    print(df_compare.head())
    df_compare.to_csv(args.output, index=False)
    print(f"Merged results written to {args.output}")
