from .external_model import get_model_from_yaml_file
from ..file_io.file_paths import get_models_path

model_names = ["ewars_Plus"]

models_path = get_models_path()
# models = {name: get_model_from_yaml_file(models_path / name / 'config.yml') for name in model_names}

models = {}
for name in model_names:
    config_path = models_path / name / "config.yml"
    if not config_path.exists():
        continue
    working_dir = models_path / name
    model = get_model_from_yaml_file(config_path, working_dir)
    models[name] = model
