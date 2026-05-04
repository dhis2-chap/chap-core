import numpy as np


def forecast_actual_value(values: list[float], output_statistic: str) -> float:
    samples = np.array(values, dtype=float)
    if output_statistic == "mean":
        return float(np.mean(samples))
    if output_statistic.startswith("q"):
        try:
            q = float(output_statistic[1:]) / 100.0
        except ValueError:
            q = 0.5
        return float(np.quantile(samples, q))
    return float(np.median(samples))
