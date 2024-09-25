from typing import Optional

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from chap_core.api_types import RequestV1
from chap_core.assessment.prediction_evaluator import Predictor
from chap_core.datatypes import Samples
from chap_core.rest_api_src.data_models import FullPredictionResponse
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


def get_app():
    app = FastAPI(root_path="/v1")
    origins = [
        "*",  # Allow all origins
        "http://localhost:3000",
        "localhost:3000",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


def samples_to_json(samples_dataset: DataSet[Samples]):
    data_values = []
    for location, samples in samples_dataset.items():
        for period, data in zip(samples.time_periods, samples.data):
            data_values.append(
                FullPredictionResponse(
                    orgunit=location, period=period.id(), data=data.tolist()
                )
            )
    return data_values


def get_rest_api(estimator):
    app = get_app()
    predictors: dict[str, Predictor] = {}
    predictions: dict[str, DataSet[Samples]] = {}

    @app.post("/train")
    def train(self, data: RequestV1, name: Optional[str] = None) -> dict:
        name = name or "model_{len(self._models)}"
        predictors[name] = estimator.train(data)
        return {"name": name}

    @app.post("/predict")
    def predict(model_name: str, data: RequestV1):
        samples: DataSet[Samples] = predictors[model_name].predict(data)
        return samples_to_json(samples)
