"""Naive predictor that estimates mean and std per location."""
import json
import numpy as np
import pandas as pd
from cyclopts import App

app = App()


@app.command()
def train(training_data_filename: str, model_path: str):
    """Train by computing mean and std of disease_cases per location."""
    df = pd.read_csv(training_data_filename)
    stats = df.groupby("location")["disease_cases"].agg(["mean", "std"]).to_dict()
    with open(model_path, "w") as f:
        json.dump(stats, f)


@app.command()
def predict(
    model_filename: str,
    historic_data_filename: str,
    future_data_filename: str,
    output_filename: str,
):
    """Predict by sampling from normal distribution based on learned stats."""
    with open(model_filename) as f:
        stats = json.load(f)

    future_df = pd.read_csv(future_data_filename)
    n_samples = 100

    rows = []
    for _, row in future_df.iterrows():
        loc = row["location"]
        mean = stats["mean"].get(loc, 0)
        std = stats["std"].get(loc, 1) or 1
        samples = np.maximum(0, np.random.normal(mean, std, n_samples))
        row_data = {"time_period": row["time_period"], "location": loc}
        row_data.update({f"sample_{i}": s for i, s in enumerate(samples)})
        rows.append(row_data)

    pd.DataFrame(rows).to_csv(output_filename, index=False)


if __name__ == "__main__":
    app()
