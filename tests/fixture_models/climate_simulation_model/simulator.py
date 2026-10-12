"""Minimal simulation model: autoregressive Poisson cases driven by lagged rainfall and temperature.

Rows that already have disease_cases are kept; missing rows are simulated in time order, so a simulation can be
continued by passing the previous output with new covariate rows appended.
"""

import sys
import zlib

import numpy as np
import pandas as pd
import yaml

DEFAULTS = {
    "intercept": 2.0,
    "rainfall_effect": 0.3,
    "temperature_effect": 0.2,
    "autoregressive_effect": 0.5,
    "location_sd": 0.5,
    "seed": 0,
}
RAINFALL_SCALE = 150.0
TEMPERATURE_REFERENCE = 25.0
TEMPERATURE_SCALE = 3.0


def rng_for(seed: int, *keys: str) -> np.random.Generator:
    """A generator that depends only on the seed and the keys, not on what else is in the dataset."""
    return np.random.default_rng([seed, *(zlib.crc32(key.encode()) for key in keys)])


def simulate_location(df: pd.DataFrame, location: str, params: dict) -> pd.Series:
    seed = int(params["seed"])
    intercept = float(params["intercept"])
    location_effect = rng_for(seed, location).normal(0.0, float(params["location_sd"]))
    rainfall = df["rainfall"].to_numpy() / RAINFALL_SCALE
    temperature = (df["mean_temperature"].to_numpy() - TEMPERATURE_REFERENCE) / TEMPERATURE_SCALE
    cases = df["disease_cases"].to_numpy(dtype=float)
    for t in range(len(df)):
        if not np.isnan(cases[t]):
            continue
        lagged_rainfall = rainfall[t - 1] if t > 0 else 0.0
        lagged_log_cases = np.log1p(cases[t - 1]) - intercept if t > 0 else 0.0
        log_rate = (
            intercept
            + location_effect
            + float(params["rainfall_effect"]) * lagged_rainfall
            + float(params["temperature_effect"]) * temperature[t]
            + float(params["autoregressive_effect"]) * lagged_log_cases
        )
        cases[t] = rng_for(seed, location, str(df["time_period"].iloc[t])).poisson(np.exp(log_rate))
    return pd.Series(cases, index=df.index)


def simulate(covariates_file: str, out_file: str, config_file: str) -> None:
    with open(config_file) as f:
        config = yaml.safe_load(f) or {}
    params = DEFAULTS | (config.get("user_option_values") or {})

    df = pd.read_csv(covariates_file, index_col=0).sort_values(["location", "time_period"])
    if "disease_cases" not in df.columns:
        df["disease_cases"] = np.nan
    simulated: list[pd.Series] = [simulate_location(group, str(location), params) for location, group in df.groupby("location")]
    df["disease_cases"] = pd.concat(simulated)
    df.to_csv(out_file, index=False)


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    if command == "simulate":
        simulate(*args)
    else:
        raise NotImplementedError(f"This is a simulation-only model; '{command}' is not supported")
