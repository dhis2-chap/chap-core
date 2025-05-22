from pathlib import Path


def get_results_path():
    path = Path(__file__).parent.parent.parent / "results"
    path.mkdir(exist_ok=True)
    return path


def get_example_data_path():
    path = Path(__file__).parent.parent.parent / "example_data"
    return path


def get_config_path():
    path = Path(__file__).parent.parent.parent / "config"
    path.mkdir(exist_ok=True)
    return path


def get_models_path():
    path = Path(__file__).parent.parent.parent / "external_models"
    path.mkdir(exist_ok=True)
    return path
