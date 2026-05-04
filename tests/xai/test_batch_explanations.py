from types import SimpleNamespace

from chap_core.xai.batch_explanations import run_explanations_task
from chap_core.xai.method_registry import NATIVE_SHAP


def test_run_explanations_task_returns_prediction_id_for_native_shap(monkeypatch):
    prediction = SimpleNamespace(id=42, forecasts=[SimpleNamespace()], dataset_id=1)
    session = SimpleNamespace(session=SimpleNamespace(get=lambda model, prediction_id: prediction))
    monkeypatch.setattr("chap_core.xai.batch_explanations.has_native_shap", lambda _prediction: True)

    result = run_explanations_task(
        prediction_id=42,
        xai_method_name=NATIVE_SHAP,
        output_statistic="median",
        top_k=10,
        session=session,
    )

    assert result == 42
