"""Minimal chapkit model for chap-core integration testing.

Uses seasonal naive prediction: for each location, predict the same
month's disease_cases value from the most recent year in training data.
"""

from typing import Any

import structlog
from geojson_pydantic import FeatureCollection

from chapkit import BaseConfig
from chapkit.api import AssessedStatus, MLServiceBuilder, MLServiceInfo, ModelMetadata, PeriodType
from chapkit.artifact import ArtifactHierarchy
from chapkit.data import DataFrame
from chapkit.ml import FunctionalModelRunner

log = structlog.get_logger()


class TestConfig(BaseConfig):
    prediction_periods: int = 3


def _extract_month(period_str: str) -> int:
    """Extract month from period string like '2023-01' or '202301'."""
    s = str(period_str)
    if "-" in s:
        return int(s.split("-")[1])
    return int(s[4:6]) if len(s) >= 6 else 1


async def on_train(
    config: TestConfig,
    data: DataFrame,
    geo: FeatureCollection | None = None,
) -> Any:
    df = data.to_pandas()
    if "disease_cases" not in df.columns:
        log.warning("no disease_cases column, returning empty model")
        return {"seasonal": {}, "fallback": 0.0}

    # Build seasonal lookup: (location, month) -> last observed value
    seasonal: dict[tuple[str, int], float] = {}
    if "time_period" in df.columns and "location" in df.columns:
        df = df.sort_values("time_period")
        for _, row in df.iterrows():
            loc = str(row["location"])
            month = _extract_month(row["time_period"])
            val = row["disease_cases"]
            if val is not None and val == val:  # skip NaN
                seasonal[(loc, month)] = float(val)

    fallback = float(df["disease_cases"].mean()) if len(df) > 0 else 0.0
    log.info(
        "trained", locations=len({k[0] for k in seasonal}), months=len({k[1] for k in seasonal}), fallback=fallback
    )
    return {"seasonal": seasonal, "fallback": fallback}


async def on_predict(
    config: TestConfig,
    model: Any,
    historic: DataFrame,
    future: DataFrame,
    geo: FeatureCollection | None = None,
) -> DataFrame:
    future_df = future.to_pandas()
    seasonal = model["seasonal"]
    fallback = model["fallback"]

    predictions = []
    for _, row in future_df.iterrows():
        loc = str(row.get("location", ""))
        month = _extract_month(row.get("time_period", "01"))
        predictions.append(seasonal.get((loc, month), fallback))

    future_df["sample_0"] = predictions
    log.info("predicted", rows=len(future_df), unique_values=len(set(predictions)))
    return DataFrame.from_pandas(future_df)


info = MLServiceInfo(
    id="chapkit-test-model",
    display_name="Chapkit Test Model",
    version="1.0.0",
    description="Minimal model for integration testing",
    model_metadata=ModelMetadata(
        author="Test",
        author_assessed_status=AssessedStatus.yellow,
    ),
    period_type=PeriodType.monthly,
    required_covariates=["population", "rainfall", "mean_temperature"],
    allow_free_additional_continuous_covariates=False,
)

hierarchy = ArtifactHierarchy(
    name="test_model",
    level_labels={0: "training", 1: "prediction"},
)

runner = FunctionalModelRunner(on_train=on_train, on_predict=on_predict)

app = (
    MLServiceBuilder(
        info=info,
        config_schema=TestConfig,
        hierarchy=hierarchy,
        runner=runner,
    )
    .with_registration(keepalive_interval=15)
    .build()
)
