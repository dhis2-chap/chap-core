from .external_model import get_model_from_yaml_file
from ..file_io.file_paths import get_models_path
model_names = ['ewars_Plus']

models_path = get_models_path()
models = {name: get_model_from_yaml_file(models_path / name / 'config.yml') for name in model_names}
