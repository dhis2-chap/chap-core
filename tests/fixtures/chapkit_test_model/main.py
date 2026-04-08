"""Minimal chapkit model for chap-core integration testing.

Returns the mean of disease_cases from training data as predictions.
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


async def on_train(
    config: TestConfig,
    data: DataFrame,
    geo: FeatureCollection | None = None,
) -> Any:
    df = data.to_pandas()
    mean_cases = float(df["disease_cases"].mean()) if "disease_cases" in df.columns else 0.0
    log.info("trained", mean_cases=mean_cases, rows=len(df))
    return {"mean_cases": mean_cases}


async def on_predict(
    config: TestConfig,
    model: Any,
    historic: DataFrame,
    future: DataFrame,
    geo: FeatureCollection | None = None,
) -> DataFrame:
    future_df = future.to_pandas()
    future_df["sample_0"] = model["mean_cases"]
    log.info("predicted", rows=len(future_df), value=model["mean_cases"])
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
    .with_registration(
        host="host.docker.internal",  # workaround: SERVICEKIT_HOST env var is ignored (see SERVICEKIT_BUGS.md)
    )
    .build()
)
