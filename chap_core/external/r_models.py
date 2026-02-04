from typing import Any

model_names = ["ewars_Plus"]
models: dict[str, Any] = {}
# for name in model_names:
#     config_path = models_path / name / "config.yml"
#     if not config_path.exists():
#         continue
#     working_dir = models_path / name
#     model = get_model_from_yaml_file(config_path, working_dir)
#     models[name] = model
