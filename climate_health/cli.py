"""Console script for climate_health."""
from climate_health.file_io.load import load_data_set
from climate_health.predictor import get_model
from .assessment.prediction_evaluator import evaluate_model

import typer


def evaluate(model_name: str, dataset_name: str):
    '''
    Evaluate a model on a dataset using forecast cross validation
    '''
    dataset = load_data_set(dataset_name)
    model = get_model(model_name)


def main_function():
    '''
    This function should just be type hinted with common types,
    and it will run as a command line function
    Simple function

    >>> main()

    '''
    return


def main():
    typer.run(main_function)
    print("Yay! You managed to run the main function!")


if __name__ == "__main__":
    main()
