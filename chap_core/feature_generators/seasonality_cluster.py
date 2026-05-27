"""Seasonality cluster feature generator.

Clusters locations by their normalized seasonal disease profiles using KMeans,
then adds a `cluster_id` column to the dataset.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from chap_core.feature_generators import FeatureGenerator, FeatureGeneratorSpec, feature_generator
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


def _extract_period_of_year(time_periods: pd.Series) -> tuple[pd.Series, int]:
    """Extract period-of-year (week or month) from a time_period Series.

    Handles both pandas Period objects and string representations.

    Returns
    -------
    tuple[pd.Series, int]
        (period_of_year series, periods_per_year)
    """
    first = time_periods.iloc[0]
    if hasattr(first, "freqstr"):
        freq = first.freqstr
        if "W" in freq:
            return time_periods.apply(lambda p: p.week), 52
        else:
            return time_periods.apply(lambda p: p.month), 12

    # String fallback
    timestamps = pd.to_datetime(time_periods)
    second = time_periods.iloc[1]
    if "W" in str(first) or "W" in str(second):
        return timestamps.dt.isocalendar().week.astype(int), 52
    return timestamps.dt.month, 12


def compute_seasonality_clusters(df: pd.DataFrame, n_clusters: int = 4) -> dict[str, int]:
    """Compute KMeans cluster labels from seasonal disease profiles.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: location, time_period, disease_cases.
    n_clusters : int
        Max number of clusters (capped by number of locations).

    Returns
    -------
    dict[str, int]
        Mapping from location to cluster label.
    """
    df = df.copy()
    df["period_of_year"], _ = _extract_period_of_year(df["time_period"])

    agg = df.groupby(["location", "period_of_year"])["disease_cases"].mean().reset_index()
    totals = agg.groupby("location")["disease_cases"].sum().reset_index()
    totals.columns = ["location", "total"]
    agg = agg.merge(totals, on="location")
    agg["normalized_cases"] = np.where(agg["total"] > 0, agg["disease_cases"] / agg["total"], 0)

    pivot = agg.pivot(index="location", columns="period_of_year", values="normalized_cases").fillna(0)
    locations = pivot.index.tolist()

    n = min(n_clusters, len(locations))
    kmeans = KMeans(n_clusters=n, random_state=0, n_init=10)
    labels = kmeans.fit_predict(pivot.values)
    return dict(zip(locations, labels, strict=True))


@feature_generator()
class SeasonalityClusterGenerator(FeatureGenerator):
    spec = FeatureGeneratorSpec(
        generator_id="seasonality_cluster",
        name="Seasonality Cluster",
        description="Clusters locations by normalized seasonal disease profiles using KMeans.",
    )

    def generate(self, dataset: DataSet) -> DataSet:
        df = dataset.to_pandas()
        cluster_map = compute_seasonality_clusters(df)
        df["cluster_id"] = df["location"].map(cluster_map).astype(float)
        return DataSet.from_pandas(df)
