# This test is for the new hpo eval pipeline using Evaluation.create
# and calculate_metrics instead of evaluate_model.

from types import SimpleNamespace

from chap_core.hpo.objective import Objective
import chap_core.hpo.objective as obj_module


def test_objective_calls_evaluation_create_and_returns_metric(monkeypatch):
    captured = {}

    fake_estimator = object()
    fake_evaluation = object()

    class FakeTemplate:
        model_template_config = SimpleNamespace(name="fake_template", version="1.0")

        def get_model(self, config):
            captured["get_model_config"] = config
            return lambda: fake_estimator  # returns a callable that returns fake_estimator when called

    class FakeBacktestParams:
        n_splits = 2
        n_periods = 3
        stride = 1

    class FakeUUID:
        hex = "a1b2c3d4e5f6g7h8"

    monkeypatch.setattr(obj_module, "uuid4", lambda: FakeUUID(), raising=True)

    def fake_create(cls, **kwargs):
        captured["create_kwargs"] = kwargs
        return fake_evaluation

    monkeypatch.setattr(
        obj_module.Evaluation,
        "create",
        classmethod(fake_create),
        raising=True,
    )

    def fake_calculate_metrics(*, evaluation, metric_ids):
        captured["metrics_args"] = {
            "evaluation": evaluation,
            "metric_ids": metric_ids,
        }
        return {"MSE": 0.42}

    monkeypatch.setattr(
        obj_module,
        "calculate_metrics",
        fake_calculate_metrics,
        raising=True,
    )

    o = Objective(
        model_template=FakeTemplate(),
        metric="MSE",
        backtest_params=FakeBacktestParams(),
    )

    score = o(config={"x": [1]}, dataset="dummy")

    assert score == 0.42
    assert captured["create_kwargs"]["estimator"] is fake_estimator
    assert captured["create_kwargs"]["dataset"] == "dummy"
    assert captured["create_kwargs"]["backtest_params"].n_splits == 2
    assert captured["create_kwargs"]["backtest_name"] == "hpo_evaluation_a1b2c3d4"
    assert captured["create_kwargs"]["historical_context_years"] == 6
    assert captured["metrics_args"] == {
        "evaluation": fake_evaluation,
        "metric_ids": ["MSE"],
    }

if __name__ == "__main__":
    import sys, pytest

    sys.exit(pytest.main([__file__]))