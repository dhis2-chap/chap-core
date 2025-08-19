from pathlib import Path
from chap_core.predictor.model_registry import ModelRegistry


def test_model_registry_from_local_config_file():
    config_file = Path(__file__).parent.parent / "config/model_templates/default.yaml"
    registry = ModelRegistry.from_model_templates_config_file(config_file)
    assert len(registry.list_specifications()) > 0
    print(registry.list_specifications())
