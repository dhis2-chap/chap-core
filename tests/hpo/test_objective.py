from chap_core.hpo.objective import Objective
import chap_core.hpo.objective as obj_module


def test_objective_calls_evaluate_model_and_returns_metric(monkeypatch):
    class FakeTemplate:
        def get_model(self, config):
            return lambda: object()

    # Patch the template constructor used in Objective.__init__
    monkeypatch.setattr(
        obj_module.ModelTemplate,
        "from_directory_or_github_url",
        classmethod(lambda cls, *a, **k: FakeTemplate()),
        raising=True,
    )

    # Patch evaluate_model to return a dict with the expected metric
    def fake_eval(model, data, prediction_length, n_test_sets):
        return [{"MSE": 0.42, "MAE": 0.2}, {"something_else": {}}]

    monkeypatch.setattr(obj_module, "evaluate_model", fake_eval, raising=True)

    o = Objective(model_template=FakeTemplate(), metric="MSE", prediction_length=3, n_splits=2)
    score = o(config={"user_option_values": {"x": [1]}}, dataset="dummy")
    assert score == 0.42


if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__]))
