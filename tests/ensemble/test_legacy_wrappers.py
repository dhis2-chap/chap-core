from chap_core.ensemble.wrappers import BaseModelSpec, TemplateWithConfig


class _DummyTemplate:
    def __init__(self):
        self.last_config = None
        self.name = "dummy"

    def get_model(self, config):
        self.last_config = config
        return "model"


def test_template_with_config_passes_config():
    template = _DummyTemplate()
    wrapper = TemplateWithConfig(template, {"alpha": 1})

    model = wrapper.get_model()

    assert model == "model"
    assert template.last_config == {"alpha": 1}


def test_base_model_spec_stores_values():
    template = _DummyTemplate()
    spec = BaseModelSpec(template=template, config=None)

    assert spec.config is None
    assert spec.template is not None
