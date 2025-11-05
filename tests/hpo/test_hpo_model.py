from chap_core.hpo.hpoModel import HpoModel


def test_hpo_model_train_selects_best_and_returns_trained(monkeypatch):
    class FakeSearcher:
        def reset(self, space):
            # two trials then stop
            self._seq = [{"x": 2}, {"x": 1}]
            self._i = 0

        def ask(self):
            if self._i >= len(self._seq):
                return None
            params = self._seq[self._i]
            self._i += 1
            return params

        def tell(self, params, result):
            pass

    class FakeEstimator:
        def __init__(self, config):
            self.config = config

        def train(self, dataset):
            return "trained-model"

    class FakeTemplate:
        def get_model(self, config):
            return FakeEstimator(config)

    class FakeObjective:
        def __init__(self):
            self.model_template = FakeTemplate()

        def __call__(self, config, dataset):
            return 1

    # Patch model_validate to return identity
    # import chap_core.hpo.hpoModel as hm_module
    # monkeypatch.setattr(hm_module.ModelConfiguration, "model_validate", lambda x: x, raising=True)

    base_cfg = {"user_option_values": {"x": [1, 2]}}
    model = HpoModel(
        searcher=FakeSearcher(),
        objective=FakeObjective(),
        direction="minimize",
        model_configuration=base_cfg,
    )
    out = model.train(dataset="dummy-dataset")
    assert out == "trained-model"
    best = model.get_best_config
    assert best["user_option_values"]["x"] == 2


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__]))
