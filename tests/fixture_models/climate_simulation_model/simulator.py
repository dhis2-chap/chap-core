"""Minimal simulation model: Poisson cases driven by lagged rainfall and temperature."""

import sys

import numpy as np
import pandas as pd
import yaml

DEFAULTS = {
    "intercept": 2.0,
    "rainfall_effect": 0.3,
    "temperature_effect": 0.2,
    "location_sd": 0.5,
    "seed": 0,
}


def standardize(series: pd.Series) -> pd.Series:
    sd = series.std()
    return (series - series.mean()) / sd if sd > 0 else series * 0.0


def simulate(covariates_file: str, out_file: str, config_file: str) -> None:
    with open(config_file) as f:
        config = yaml.safe_load(f) or {}
    params = DEFAULTS | (config.get("user_option_values") or {})
    rng = np.random.default_rng(int(params["seed"]))

    df = pd.read_csv(covariates_file, index_col=0).sort_values(["location", "time_period"])
    rainfall = standardize(df["rainfall"])
    lagged_rainfall = rainfall.groupby(df["location"]).shift(1).fillna(0.0)
    temperature = standardize(df["mean_temperature"])

    locations = df["location"].unique()
    location_effect = dict(zip(locations, rng.normal(0.0, float(params["location_sd"]), len(locations))))

    log_rate = (
        float(params["intercept"])
        + df["location"].map(location_effect)
        + float(params["rainfall_effect"]) * lagged_rainfall
        + float(params["temperature_effect"]) * temperature
    )
    df["disease_cases"] = rng.poisson(np.exp(log_rate))
    df.to_csv(out_file, index=False)


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    if command == "simulate":
        simulate(*args)
    else:
        raise NotImplementedError(f"This is a simulation-only model; '{command}' is not supported")
