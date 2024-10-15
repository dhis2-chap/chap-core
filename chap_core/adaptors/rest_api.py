from typing import List

import pydantic
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from chap_core.datatypes import remove_field
from chap_core.model_spec import get_dataclass
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
import logging

logger = logging.getLogger(__name__)

def generate_app(estimator, working_dir: str):
    app = FastAPI()
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

    dc = get_dataclass(estimator)
    model = pydantic.create_model('TrainingData', **dc.__annotations__)
    training_data_filename = f"{working_dir}/training_data.csv"
    model_path = f"{working_dir}/model"

    @app.command()
    def train(training_data: List[model]):
        '''
        Train a model using historic data

        Parameters
        ----------
        training_data_filename: str
            The path to the training data file
        model_path: str
            The path to save the trained model
        '''
        logger.info(f"Loading data from {training_data_filename} as {dc}")
        dataset = DataSet.df_from_pydantic_observations()
        predictor = estimator.train(dataset)
        predictor.save(model_path)

    @app.command()
    def predict(model_filename: str, historic_data_filename: str, future_data_filename: str, output_filename: str):
        '''
        Predict using a trained model

        Parameters
        ----------
        model_filename: str
            The path to the model file trained with the train command
        historic_data_filename: str
            The path to the historic data file, i.e. real data up to the present/prediction start
        future_data_filename: str
            The path to the future data file, i.e. forecasted predictors for the future
        '''
        dataset = DataSet.from_csv(historic_data_filename, dc)
        future_dc = remove_field(dc, 'disease_cases')
        future_data = DataSet.from_csv(future_data_filename, future_dc)
        predictor = estimator.load_predictor(model_filename)
        forecasts = predictor.predict(dataset, future_data)
        forecasts.to_csv(output_filename)

    return app