import argparse
import json
import re
import pandas as pd
from chap_core.database.tables import BackTestForecast

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

def _inject_aliases_for_backtest(d: dict) -> dict:
    """
    After snake-casing, also provide the camelCase aliases that the Pydantic model
    may expect (useful when populate_by_name is not enabled).
    This keeps the snake_case keys AND adds alias keys.
    """
    # Top-level date/time fields
    if "start_date" in d and "startDate" not in d:
        d["startDate"] = d["start_date"]
    if "end_date" in d and "endDate" not in d:
        d["endDate"] = d["end_date"]
    # timestamp is often the same in both styles; include just in case
    if "timestamp" in d and "timestamp" not in d:
        d["timestamp"] = d["timestamp"]

    # Forecast-level fields (common one is periodId)
    if isinstance(d.get("forecasts"), list):
        for i, item in enumerate(d["forecasts"]):
            if isinstance(item, dict):
                if "period_id" in item and "periodId" not in item:
                    item["periodId"] = item["period_id"]
                # add more per-forecast aliases here if needed
                d["forecasts"][i] = item
    return d

def load_forecast(path: str) -> pd.DataFrame:
    with open(path) as f:
        raw_data = json.load(f)

    # 1) Convert all keys to snake_case
    snake = convert_keys(raw_data)

    # 2) Ensure model expects estimator_id (your code already does this)
    snake["estimator_id"] = snake.pop("model_id")

    # 3) ALSO provide alias keys the model may require (startDate, endDate, periodId)
    snake = _inject_aliases_for_backtest(snake)

    print("Top-level keys:", list(snake.keys()))

    # 4) Validate / parse into Pydantic model
    backtest = BackTestForecast.sqlsmodel_validate(snake)

    # Debug: confirm fields exist and are populated
    print(type(backtest))
    print(backtest.model_dump().keys())
    print(backtest.model_fields.keys())
    print("timestamp:", getattr(backtest, "timestamp", None))
    print("start_date:", getattr(backtest, "start_date", None))
    print("end_date:", getattr(backtest, "end_date", None))

    # 5) Expand nested forecasts into flat records
    records = []
    for f in backtest.forecasts:
        period = getattr(f, "period_id", None)  # after validation it should be present
        for step, val in enumerate(f.values, start=1):
            records.append(
                {
                    "period": period,
                    "step": step,
                    "predicted": val,
                }
            )

    # 6) Build DataFrame and parse dates
    df_pred = pd.DataFrame(records)
    df_pred["period_dt"] = pd.to_datetime(df_pred["period"], format="%Y%m", errors="coerce")
    return df_pred

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

    # Load dataframe
    df_pred = load_forecast(args.pred)