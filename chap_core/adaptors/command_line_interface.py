from cyclopts import App

from chap_core.datatypes import remove_field, create_tsdataclass
from chap_core.model_spec import get_dataclass
from chap_core.models import ModelTemplate
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
import logging

logger = logging.getLogger(__name__)


# TODO write an app for a model template
def generate_app(estimator):
    app = App()
    dc = get_dataclass(estimator)

    @app.command()
    def train(training_data_filename: str, model_path: str):
        """
        Train a model using historic data

        Parameters
        ----------
        training_data_filename: str
            The path to the training data file
        model_path: str
            The path to save the trained model
        """
        logger.info(f"Loading data from {training_data_filename} as {dc}")
        dataset = DataSet.from_csv(training_data_filename, dc)
        predictor = estimator.train(dataset)
        predictor.save(model_path)

    @app.command()
    def predict(model_filename: str, historic_data_filename: str, future_data_filename: str, output_filename: str):
        """
        Predict using a trained model

        Parameters
        ----------
        model_filename: str
            The path to the model file trained with the train command
        historic_data_filename: str
            The path to the historic data file, i.e. real data up to the present/prediction start
        future_data_filename: str
            The path to the future data file, i.e. forecasted predictors for the future
        """
        dataset = DataSet.from_csv(historic_data_filename, dc)
        future_dc = remove_field(dc, "disease_cases")
        future_data = DataSet.from_csv(future_data_filename, future_dc)
        predictor = estimator.load_predictor(model_filename)
        forecasts = predictor.predict(dataset, future_data)
        forecasts.to_csv(output_filename)

    return app


def generate_template_app(model_template: ModelTemplate):
    app = App()

    @app.command()
    def train(training_data_filename: str, model_path: str, model_config_path: str = None):
        """
        Train a model using historic data

        Parameters
        ----------
        training_data_filename: str
            The path to the training data file
        model_path: str
            The path to save the trained model
        """
        model_config = model_template.get_config_class().parse_file(model_config_path)

        # TODO: create method in ModelTemplate to get the actual fields
        # Or give the model the responsibilitiy
        estimator = model_template.get_model(model_config)
        data_fields = estimator.covariate_names
        dc = create_tsdataclass(data_fields)
        dataset = DataSet.from_csv(training_data_filename, dc)

        predictor = estimator.train(dataset)
        predictor.save(model_path)

    # TODO: send in model config again here
    @app.command()
    def predict(model_filename: str, historic_data_filename: str,
                future_data_filename: str, output_filename: str, model_config_path: str = None):
        """
        Predict using a trained model

        Parameters
        ----------
        training_data_filename: str
            The path to the training data file
        model_filename: str
            The path to the model file trained with the train command
        historic_data_filename: str
            The path to the historic data file, i.e. real data up to the present/prediction start
        future_data_filename: str
            The path to the future data file, i.e. forecasted predictors for the future
        """
        model_config = model_template.get_config_class().parse_file(model_config_path)


        estimator = model_template.get_model(model_config)
        dc = create_tsdataclass(estimator.covariate_names)
        future_dc = remove_field(dc, "disease_cases")
        predictor = estimator.load_predictor(model_filename)

        dataset = DataSet.from_csv(historic_data_filename, dc)
        future_data = DataSet.from_csv(future_data_filename, future_dc)

        forecasts = predictor.predict(dataset, future_data)
        forecasts.to_csv(output_filename)
    #
    # model_template_config = ModelTemplateConfig(
    #     name=model_template.name,
    #     entry_points=EntryPointConfig(
    #         train=CommandConfig(command='python main.py train {train_data} {model} {model_config}',
    #                             parameters={'train_data': 'str', 'model': 'str', 'model_config': 'str'}),
    #         predict=CommandConfig("python main.py predict {model} {historic_data} {future_data} {out_file} {model_config}",
    #                               parameters={n: 'str' for n in ['historic_data', 'future_data', 'out_file', 'model_config']})),
    #
    #
    #
    #
    #
    #
    #
    #
    #     )

    #)

    return app, train, predict