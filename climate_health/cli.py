"""Console script for climate_health."""
import webbrowser
from pathlib import PurePath
from typing import Literal
from cyclopts import App
from climate_health.predictor import get_model, models, ModelType
from climate_health.file_io.example_data_set import datasets, DataSetType
from .assessment.prediction_evaluator import evaluate_model

app = App()

@app.command()
def evaluate(model_name: ModelType, dataset_name: DataSetType, max_splits: int):
    '''
    Evaluate a model on a dataset using forecast cross validation
    '''
    dataset = datasets[dataset_name].load()
    model = get_model(model_name)()
    results, table = evaluate_model(dataset, model, max_splits, start_offset=24, return_table=True)
    output_filename= PurePath(f'./{model_name}_{dataset_name}_results.html')
    table_filename = output_filename.with_suffix('.csv')
    results.save(output_filename)
    table.to_csv(table_filename)
    webbrowser.open(output_filename)

def main_function():
    '''
    This function should just be type hinted with common types,
    and it will run as a command line function
    Simple function

    >>> main()

    '''
    return


def main():
    app()
    # typer.run(evaluate)
    print("Yay! You managed to run the main function!")


if __name__ == "__main__":
    app()
