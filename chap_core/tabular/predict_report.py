"""HTML performance report for ``chap tabular predict``.

Only produced when the scored dataset carries the target column. For a
classifier it shows the confusion matrix and a ROC curve (or, without
probabilities, a per-class metric bar chart); for a regressor a predicted vs
actual scatter. All numbers are external test numbers, not cross-validation.
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.figure import Figure

    from chap_core.tabular.predict import PredictionResult


def _png_data_uri(fig: Figure) -> str:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    buffer = io.BytesIO()
    FigureCanvasAgg(fig).print_png(buffer)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _metrics_table(performance: dict[str, float]) -> str:
    rows = "".join(f"<tr><td>{html.escape(name)}</td><td>{value:.4f}</td></tr>" for name, value in performance.items())
    return f"<table><tbody>{rows}</tbody></table>"


def _confusion_matrix_table(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    from sklearn.metrics import confusion_matrix

    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return f"""<table class="cm">
<thead><tr><th></th><th colspan="2">predicted</th></tr>
<tr><th></th><th>0</th><th>1</th></tr></thead>
<tbody>
<tr><th>actual 0</th><td class="ok">{tn}<br><small>TN</small></td><td class="bad">{fp}<br><small>FP</small></td></tr>
<tr><th>actual 1</th><td class="bad">{fn}<br><small>FN</small></td><td class="ok">{tp}<br><small>TP</small></td></tr>
</tbody></table>"""


def _roc_plot(y_true: np.ndarray, y_score: np.ndarray) -> str:
    from matplotlib.figure import Figure
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig = Figure(figsize=(5, 5))
    ax = fig.subplots()
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _png_data_uri(fig)


def _per_class_metric_plot(performance: dict[str, float]) -> str:
    from matplotlib.figure import Figure

    names = ["precision", "recall", "f1"]
    values = [performance[name] for name in names]

    fig = Figure(figsize=(5, 4))
    ax = fig.subplots()
    ax.bar(names, values, color="#4c78a8")
    ax.set_ylim(0, 1)
    ax.set_title("Precision / recall / F1")
    for index, value in enumerate(values):
        ax.text(index, value + 0.02, f"{value:.2f}", ha="center")
    fig.tight_layout()
    return _png_data_uri(fig)


def _predicted_vs_actual_plot(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5, 5))
    ax = fig.subplots()
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="none")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="grey", label="y = x")
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
    ax.set_title("Predicted vs actual")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _png_data_uri(fig)


def render_prediction_report(result: PredictionResult) -> str:
    """Return the prediction performance report as an HTML string."""
    if result.performance is None or result.target_name is None:
        raise ValueError("A prediction report needs the target column in the scored dataset.")

    frame = result.frame
    y_true = frame[result.target_name].to_numpy()
    y_pred = frame["prediction"].to_numpy()

    manifest = {
        "Task": result.task,
        "Target": result.target_name,
        "Rows scored": len(frame),
    }
    manifest_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>" for k, v in manifest.items()
    )

    if result.task == "classification":
        sections = f"<h2>Confusion matrix</h2>{_confusion_matrix_table(y_true, y_pred)}"
        if result.has_probability:
            plot = _roc_plot(y_true, frame["probability"].to_numpy())
            caption = "ROC curve"
        else:
            plot = _per_class_metric_plot(result.performance)
            caption = "Per-class metrics (no probabilities available)"
    else:
        sections = ""
        plot = _predicted_vs_actual_plot(y_true, y_pred)
        caption = "Predicted vs actual"

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Prediction performance: {html.escape(result.target_name)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 900px; }}
table {{ border-collapse: collapse; margin: 1rem 0; }}
th, td {{ border: 1px solid #ccc; padding: 0.3rem 0.6rem; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
table.cm td {{ text-align: center; min-width: 4rem; }}
table.cm td.ok {{ background: #e7f3e7; }}
table.cm td.bad {{ background: #fdecec; }}
small {{ color: #666; }}
.note {{ background: #fff3cd; padding: 0.6rem 1rem; border-left: 4px solid #e0a800; }}
img {{ margin: 1rem 0; }}
</style>
</head>
<body>
<h1>Prediction performance</h1>
<p class="note">All numbers are external test numbers computed on the scored dataset, not cross-validation.</p>
<h2>Manifest</h2>
<table><tbody>{manifest_rows}</tbody></table>
<h2>Metrics</h2>
{_metrics_table(result.performance)}
{sections}
<h2>{html.escape(caption)}</h2>
<img src="{plot}" alt="{html.escape(caption)}">
</body>
</html>
"""


def write_prediction_report(result: PredictionResult, path: str | Path) -> None:
    Path(path).write_text(render_prediction_report(result), encoding="utf-8")
